import numpy as np
import torch
import torch.nn as nn

class NLSTMCell(nn.Module):
   
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.outer = nn.Linear(input_size + hidden_size, 4 * hidden_size)

        self.inner = nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)

    def forward(self, x_t, state):
        """
        x_t: (B, input_size)
        state: (h, c, nc) each (B, hidden_size)
        """
        h_prev, c_prev, nc_prev = state

        z = self.outer(torch.cat([x_t, h_prev], dim=-1))
        i, f, o, g = torch.chunk(z, 4, dim=-1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        inner_in = i * g

        inner_h_prev = f * c_prev

        c_t, nc_t = self.inner(inner_in, (inner_h_prev, nc_prev))

        # Outer hidden
        h_t = o * torch.tanh(c_t)

        return h_t, (h_t, c_t, nc_t)


class NLSTM(nn.Module):
  
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        cells = []
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size
            cells.append(NLSTMCell(in_size, hidden_size))
        self.cells = nn.ModuleList(cells)

    def forward(self, x):
        """
        x: (B, T, input_size)
        returns:
          h_last: (B, hidden_size) from top layer
        """
        B, T, _ = x.shape

        # init states per layer
        states = []
        for _ in range(self.num_layers):
            h0 = torch.zeros(B, self.hidden_size, device=x.device)
            c0 = torch.zeros(B, self.hidden_size, device=x.device)
            nc0 = torch.zeros(B, self.hidden_size, device=x.device)
            states.append((h0, c0, nc0))

        # time loop
        for t in range(T):
            inp = x[:, t, :]
            new_states = []
            for layer, cell in enumerate(self.cells):
                h_out, st_out = cell(inp, states[layer])
                new_states.append(st_out)
                inp = h_out  # feed to next layer
            states = new_states

        h_last = states[-1][0]
        return h_last

class NLSTMBaseline(nn.Module):
 
    def __init__(self, input_feat=1, hidden_size=10, num_layers=1, output_len=12):
        super().__init__()
        self.encoder = NLSTM(input_size=input_feat, hidden_size=hidden_size, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_len)
        )

    def forward(self, x_seq):
       
        h_last = self.encoder(x_seq)
        return self.head(h_last)

def prepare_nlstm_data(X, Y):
   
    if X.dim() == 3:
        X = X.unsqueeze(-1)
    if Y.dim() == 3:
        Y = Y.unsqueeze(-1)

    B, T_in, N, F = X.shape
    assert F == 1, f"Expected feature dim=1, got {F}"
    T_out = Y.shape[1]

    X = X.squeeze(-1)
    Y = Y.squeeze(-1)

    X = X.permute(0, 2, 1).contiguous()
    Y = Y.permute(0, 2, 1).contiguous()

    # (B*N, T_in, 1)
    X_seq = X.view(B * N, T_in).unsqueeze(-1)

    # (B*N, T_out)
    Y_out = Y.view(B * N, T_out)

    return X_seq, Y_out


def train_nlstm(model, train_loader, test_loader,device, epochs=50, lr=1e-3,loss_fn = nn.L1Loss()):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_train = 0.0

        for Xb, Yb in train_loader:
            Xb = Xb.to(device)
            Yb = Yb.to(device)

            X_seq, Y_out = prepare_nlstm_data(Xb, Yb)

            optimizer.zero_grad(set_to_none=True)
            pred = model(X_seq)
            loss = loss_fn(pred, Y_out)
            loss.backward()
            optimizer.step()

            total_train += loss.item()

        train_mae = total_train / max(1, len(train_loader))

       # eval
        model.eval()
        total_mae = 0.0
        total_mse = 0.0
        total_mape = 0.0

        mse_fn = torch.nn.MSELoss(reduction="mean")
        eps = 1e-5

        with torch.no_grad():
            for Xb, Yb in test_loader:
                Xb = Xb.to(device)
                Yb = Yb.to(device)

                X_seq, Y_out = prepare_nlstm_data(Xb, Yb)
                pred = model(X_seq)

                # MAE
                total_mae += loss_fn(pred, Y_out).item()

                # MSE
                total_mse += mse_fn(pred, Y_out).item()

                # MAPE (safe)
                batch_mape = torch.mean(
                    torch.abs((Y_out - pred) / (Y_out + eps))
                ) * 100.0
                total_mape += batch_mape.item()

        test_mae = total_mae / max(1, len(test_loader))
        test_mse = total_mse / max(1, len(test_loader))
        test_mape = total_mape / max(1, len(test_loader))

        print(
            f"Epoch {epoch:03d} | "
            f"Train MAE: {train_mae:.4f} | "
            f"Test MAE: {test_mae:.4f} | "
            f"Test MSE: {test_mse:.4f} | "
            f"Test MAPE: {test_mape:.2f}%"
        )
    return model, test_mae, test_mse, test_mape


@torch.no_grad()
def show_one_prediction(model, test_loader,device, sample_id=0, node_id=0):
    model.eval()
    Xb, Yb = next(iter(test_loader))
    Xb = Xb.to(device)
    Yb = Yb.to(device)

    X_seq, Y_out = prepare_nlstm_data(Xb, Yb)
    pred = model(X_seq)  

    B, T_in, N, _ = Xb.shape
    idx = sample_id * N + node_id

    print(f"Example prediction | sample={sample_id}, node={node_id}")
    print("GT  :", Y_out[idx].cpu().numpy())
    print("Pred:", pred[idx].cpu().numpy())





# show_prediction_with_history(
#     model=nlstm_model,
#     test_loader=test_loader,
#     device=DEVICE,
#     prepare_fn=NLSTM.prepare_nlstm_data,  # <-- key difference
#     sample_id=50,                         # global test-set index
#     node_id=200,
#     feature_in=0,
#     title="NLSTM Prediction (Past + Future)",
# )