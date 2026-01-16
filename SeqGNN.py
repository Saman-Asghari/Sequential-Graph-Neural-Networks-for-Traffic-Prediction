
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class GNBlock(nn.Module):
  
    def __init__(self, node_dim, edge_dim, global_dim, hidden_dim):
        super().__init__()

        # Edge update: [e, v_r, v_s, u] -> e'
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + 2 * node_dim + global_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, edge_dim),
        )

        # Node update: [sum_e', v, u] -> v'
        self.node_mlp = nn.Sequential(
            nn.Linear(edge_dim + node_dim + global_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, node_dim),
        )

        # Global update: [sum_e', sum_v', u] -> u'
        self.global_mlp = nn.Sequential(
            nn.Linear(edge_dim + node_dim + global_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, global_dim),
        )

    def forward(self, x, edge_attr, u, edge_index):
        
        assert x.dim() == 3 and edge_attr.dim() == 3 and u.dim() == 2
        B, N, node_dim = x.shape
        _, E, edge_dim = edge_attr.shape

        src, dst = edge_index
        src = src.to(x.device).long()
        dst = dst.to(x.device).long()

        sender_nodes = x.index_select(1, src)    # (B, E, node_dim)
        receiver_nodes = x.index_select(1, dst)  # (B, E, node_dim)

        u_edges = u.unsqueeze(1).expand(B, E, u.size(1))

        edge_in = torch.cat([edge_attr, receiver_nodes, sender_nodes, u_edges], dim=-1)
        e_prime = self.edge_mlp(edge_in)  # (B, E, edge_dim)

        aggr_edges = torch.zeros(B, N, edge_dim, device=x.device, dtype=x.dtype)

        dst_index = dst.view(1, E, 1).expand(B, E, edge_dim)
        aggr_edges.scatter_add_(dim=1, index=dst_index, src=e_prime)

        u_nodes = u.unsqueeze(1).expand(B, N, u.size(1))

        node_in = torch.cat([aggr_edges, x, u_nodes], dim=-1)
        v_prime = self.node_mlp(node_in)  # (B, N, node_dim)

        edge_agg_global = e_prime.sum(dim=1)  # (B, edge_dim)
        node_agg_global = v_prime.sum(dim=1)  # (B, node_dim)

        global_in = torch.cat([edge_agg_global, node_agg_global, u], dim=-1)
        u_prime = self.global_mlp(global_in)  # (B, global_dim)

        return v_prime, e_prime, u_prime


class GRNNBlock(nn.Module):
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.gn_core = GNBlock(
            node_dim=hidden_dim * 2,
            edge_dim=hidden_dim * 2,
            global_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
        )

    @staticmethod
    def _split_last(t):
        half = t.size(-1) // 2
        return t[..., :half], t[..., half:]

    def forward(self, G_x, G_h, edge_index):
        x_in, e_in, u_in = G_x
        x_h,  e_h,  u_h  = G_h

        x_merge = torch.cat([x_in, x_h], dim=-1)
        e_merge = torch.cat([e_in, e_h], dim=-1)
        u_merge = torch.cat([u_in, u_h], dim=-1)

        x_out, e_out, u_out = self.gn_core(x_merge, e_merge, u_merge, edge_index)

        x_y, x_h_new = self._split_last(x_out)
        e_y, e_h_new = self._split_last(e_out)
        u_y, u_h_new = self._split_last(u_out)

        return (x_y, e_y, u_y), (x_h_new, e_h_new, u_h_new)


class SeqGNN(nn.Module):
    def __init__(
        self,
        num_static_node_feats: int,
        num_static_edge_feats: int,
        hidden_dim: int,
        input_seq_len: int,
        output_seq_len: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len

        self.node_encoder = nn.Linear(1 + num_static_node_feats, hidden_dim)
        self.edge_encoder = nn.Linear(num_static_edge_feats, hidden_dim)

        # Learnable initial global vector
        self.u0 = nn.Parameter(torch.zeros(1, hidden_dim))

        self.grnn = GRNNBlock(hidden_dim)
        self.decoder_head = nn.Linear(hidden_dim, 1)

    def forward(self, x_seq, static_node, static_edge, edge_index, y_truth=None, teacher_forcing=0.0):
        """
        x_seq: (B, T_in, N, 1)
        y_truth: (B, T_out, N, 1) optional (teacher forcing)
        static_node: (N, F_n)
        static_edge: (E, F_e)
        edge_index: (2, E)
        """
        B, T_in, N, F = x_seq.shape
        assert F == 1, f"Expected speed feature dim=1, got {F}"
        assert T_in == self.input_seq_len, f"Expected T_in={self.input_seq_len}, got {T_in}"

        E = static_edge.shape[0]
        dev = x_seq.device

        static_node_b = static_node.unsqueeze(0).expand(B, -1, -1)
        static_edge_b = static_edge.unsqueeze(0).expand(B, -1, -1)

        h_node = torch.zeros(B, N, self.hidden_dim, device=dev)
        h_edge = torch.zeros(B, E, self.hidden_dim, device=dev)
        h_global = self.u0.expand(B, -1)
        G_h = (h_node, h_edge, h_global)

        for t in range(self.input_seq_len):
            speed_t = x_seq[:, t, :, :]
            node_in = torch.cat([speed_t, static_node_b], dim=-1)

            x_emb = self.node_encoder(node_in)
            e_emb = self.edge_encoder(static_edge_b)
            u_emb = self.u0.expand(B, -1)

            _, G_h = self.grnn((x_emb, e_emb, u_emb), G_h, edge_index)

        outputs = []
        last_speed = x_seq[:, self.input_seq_len - 1, :, :]

        for t in range(self.output_seq_len):
            node_in = torch.cat([last_speed, static_node_b], dim=-1)
            x_emb = self.node_encoder(node_in)
            e_emb = self.edge_encoder(static_edge_b)
            u_emb = self.u0.expand(B, -1)

            G_y, G_h = self.grnn((x_emb, e_emb, u_emb), G_h, edge_index)
            pred_speed = self.decoder_head(G_y[0])
            outputs.append(pred_speed)

            if (y_truth is not None) and (teacher_forcing > 0.0) and self.training:
                use_tf = (torch.rand(B, device=dev) < teacher_forcing).float().view(B, 1, 1)
                gt = y_truth[:, t, :, :]
                last_speed = use_tf * gt + (1.0 - use_tf) * pred_speed
            else:
                last_speed = pred_speed

        return torch.stack(outputs, dim=1)  # (B, T_out, N, 1)
def train_model(
    model: nn.Module,
    train_loader,
    test_loader,
    static_node: torch.Tensor,
    static_edge: torch.Tensor,
    edge_index: torch.Tensor,
    device: torch.device,
    learning_rate: float,
    epochs: int,
    grad_clip_norm: float,
    criterion: nn.Module | None = None,
):
    if criterion is None:
        criterion = nn.L1Loss()  # MAE

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,    
        patience=5,     
        threshold=1e-4,
    )

    # Early stopping
    early_stop_patience = 15
    min_delta = 1e-4
    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    static_node = static_node.to(device)
    static_edge = static_edge.to(device)
    edge_index = edge_index.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for Xb, Yb in train_loader:
            Xb = Xb.to(device)
            Yb = Yb.to(device)

            optimizer.zero_grad(set_to_none=True)

            preds = model(Xb, static_node, static_edge, edge_index, y_truth=None, teacher_forcing=0.0)
            loss = criterion(preds, Yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            total_loss += loss.item()

        train_mae = total_loss / max(1, len(train_loader))

        model.eval()
        val_mae = 0.0
        val_mse = 0.0
        val_mape = 0.0

        eps = 1e-5 

        with torch.no_grad():
            for Xb, Yb in test_loader:
                Xb = Xb.to(device)
                Yb = Yb.to(device)

                preds = model(
                    Xb,
                    static_node,
                    static_edge,
                    edge_index,
                    y_truth=None,
                    teacher_forcing=0.0,
                )

                # MAE (criterion = L1Loss)
                val_mae += criterion(preds, Yb).item()

                # MSE
                val_mse += torch.mean((preds - Yb) ** 2).item()

                # MAPE (safe)
                batch_mape = torch.mean(
                    torch.abs((Yb - preds) / (Yb + eps))
                ) * 100.0
                val_mape += batch_mape.item()

        denom = max(1, len(test_loader))

        val_mae /= denom
        val_mse /= denom
        val_rmse = np.sqrt(val_mse)
        val_mape /= denom

        scheduler.step(val_mae)

        print(
            f"Epoch {epoch:03d} | Train MAE: {train_mae:.4f} | "
            f"Test MAE: {val_mae:.4f} | Test RMSE: {val_rmse:.4f}"
        )

        # Early stopping check (on val_mae)
        if val_mae < (best_val - min_delta):
            pass
        else:
            bad_epochs += 1
            if bad_epochs >= early_stop_patience:
                print(f"Early stopping at epoch {epoch:03d} (best Test MAE: {best_val:.4f})")
                break

   
    return model, val_mae, val_mse, val_mape
