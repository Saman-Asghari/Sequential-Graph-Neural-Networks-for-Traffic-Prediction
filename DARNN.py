import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def prepare_darnn_data(X: torch.Tensor, Y: torch.Tensor):
    """
    X: (B, T_in, N, F) or (B, T_in, N)
    Y: (B, T_out, N, 1) or (B, T_out, N)

    returns:
      X_seq: (B*N, T_in, F)
      Y_seq: (B*N, T_out, 1)
    """
    if X.dim() == 3:
        X = X.unsqueeze(-1)
    if Y.dim() == 3:
        Y = Y.unsqueeze(-1)

    assert X.dim() == 4 and Y.dim() == 4, f"Expected 4D tensors. Got X={X.shape}, Y={Y.shape}"
    B, T_in, N, F = X.shape
    T_out = Y.shape[1]
    assert F >= 1, f"Expected at least 1 feature, got {F}"
    assert Y.shape[-1] == 1, f"Expected Y last dim=1, got {Y.shape[-1]}"

    X_seq = X.permute(0, 2, 1, 3).contiguous().view(B * N, T_in, F)
    Y_seq = Y.permute(0, 2, 1, 3).contiguous().view(B * N, T_out, 1)
    return X_seq, Y_seq


class DARNNEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.attn = nn.Linear(2 * hidden_dim + 1, 1)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
    
        B, T, F = x.shape

        h = torch.zeros(self.num_layers, B, self.hidden_dim, device=x.device)
        c = torch.zeros(self.num_layers, B, self.hidden_dim, device=x.device)

        H_list = []

        for t in range(T):
            h_prev = h[-1]  # (B, hidden)
            c_prev = c[-1]  # (B, hidden)
            x_t = x[:, t, :]  # (B, F)

            if F == 1:
                alpha = torch.ones(B, 1, device=x.device)
            else:
                scores = []
                for k in range(F):
                    x_tk = x_t[:, k:k+1]                       # (B, 1)
                    inp = torch.cat([h_prev, c_prev, x_tk], -1) # (B, 2H+1)
                    scores.append(self.attn(inp))               # (B, 1)
                scores = torch.cat(scores, dim=1)              # (B, F)
                alpha = torch.softmax(scores, dim=1)           # (B, F)

            x_tilde = alpha * x_t                               # (B, F)
            out, (h, c) = self.lstm(x_tilde.unsqueeze(1), (h, c))# (B, 1, hidden)
            H_list.append(out)

        H = torch.cat(H_list, dim=1)  # (B, T, hidden)
        return H


class DARNNDecoder(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.temporal_attn = nn.Linear(2 * hidden_dim, 1)

        self.lstm = nn.LSTM(
            input_size=1 + hidden_dim,  # [y_prev, context]
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, H_enc: torch.Tensor, y_true: torch.Tensor | None = None,
                out_len: int = 12, teacher_forcing: float = 1.0) -> torch.Tensor:
      
        B, T_in, H = H_enc.shape
        assert H == self.hidden_dim

        d = torch.zeros(self.num_layers, B, self.hidden_dim, device=H_enc.device)
        s = torch.zeros(self.num_layers, B, self.hidden_dim, device=H_enc.device)

        y_prev = torch.zeros(B, 1, device=H_enc.device)  # (B,1)
        preds = []

        for t in range(out_len):
            d_prev = d[-1] 

            scores = []
            for i in range(T_in):
                h_i = H_enc[:, i, :]                       
                inp = torch.cat([d_prev, h_i], dim=-1)    
                scores.append(self.temporal_attn(inp))     
            scores = torch.cat(scores, dim=1)            
            beta = torch.softmax(scores, dim=1)            

            context = torch.sum(beta.unsqueeze(-1) * H_enc, dim=1)  # (B, hidden)

            dec_in = torch.cat([y_prev, context], dim=-1).unsqueeze(1)  # (B,1,1+hidden)
            out, (d, s) = self.lstm(dec_in, (d, s))
            d_t = out[:, 0, :]  # (B, hidden)

            y_t = self.fc(torch.cat([d_t, context], dim=-1))  # (B,1)
            preds.append(y_t.unsqueeze(1))                    # (B,1,1)

            if y_true is not None and teacher_forcing > 0.0:
                # per-sample TF decision
                use_tf = (torch.rand(B, device=H_enc.device) < teacher_forcing).unsqueeze(-1)  # (B,1)
                y_prev = torch.where(use_tf, y_true[:, t, :], y_t)  # (B,1)
            else:
                y_prev = y_t

        return torch.cat(preds, dim=1)  # (B,T_out,1)


class DARNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, enc_layers: int = 1, dec_layers: int = 1):
        super().__init__()
        self.encoder = DARNNEncoder(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=enc_layers)
        self.decoder = DARNNDecoder(hidden_dim=hidden_dim, num_layers=dec_layers)

    def forward(self, x_seq: torch.Tensor, y_true: torch.Tensor | None = None,
                out_len: int = 12, teacher_forcing: float = 1.0) -> torch.Tensor:
        H_enc = self.encoder(x_seq)
        return self.decoder(H_enc, y_true=y_true, out_len=out_len, teacher_forcing=teacher_forcing)


def train_one_epoch_darnn(model, train_loader, optimizer, loss_fn, device,
                          teacher_forcing: float = 1.0, grad_clip_norm: float = 1.0):
    model.train()
    total_loss = 0.0

    for Xb, Yb in train_loader:
        Xb = Xb.to(device)
        Yb = Yb.to(device)

        X_seq, Y_seq = prepare_darnn_data(Xb, Yb)

        optimizer.zero_grad(set_to_none=True)
        Y_pred = model(X_seq, y_true=Y_seq, out_len=Y_seq.size(1), teacher_forcing=teacher_forcing)
        loss = loss_fn(Y_pred, Y_seq)
        loss.backward()

        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(1, len(train_loader))

@torch.no_grad()
def evaluate_darnn(model, test_loader, loss_fn, device, eps=1e-5):
    """
    Returns:
      mae, mse, mape   (mape in %)
    """
    model.eval()
    total_mae = 0.0
    total_mse = 0.0
    total_mape = 0.0

    mse_fn = torch.nn.MSELoss(reduction="mean")

    for Xb, Yb in test_loader:
        Xb = Xb.to(device)
        Yb = Yb.to(device)

        X_seq, Y_seq = prepare_darnn_data(Xb, Yb)

        Y_pred = model(
            X_seq,
            y_true=None,
            out_len=Y_seq.size(1),
            teacher_forcing=0.0,
        )

        total_mae += loss_fn(Y_pred, Y_seq).item()

        total_mse += mse_fn(Y_pred, Y_seq).item()

        batch_mape = torch.mean(
            torch.abs((Y_seq - Y_pred) / (Y_seq + eps))
        ) * 100.0
        total_mape += batch_mape.item()

    denom = max(1, len(test_loader))
    return (
        total_mae / denom,
        total_mse / denom,
        total_mape / denom,
    )


def train_darnn(
    train_loader,
    test_loader,
    device,
    hidden_size: int = 10,
    num_layers: int = 1,
    lr: float = 1e-4,
    epochs: int = 150,
    teacher_forcing: float = 1.0,
    grad_clip_norm: float = 1.0,
):
   
    sample_X, _ = next(iter(train_loader))
    F = sample_X.shape[-1] if sample_X.dim() == 4 else 1

    model = DARNN(input_dim=F, hidden_dim=hidden_size, enc_layers=num_layers, dec_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()  # MAE


    for epoch in range(1, epochs + 1):
        train_mae = train_one_epoch_darnn(
            model, train_loader, optimizer, loss_fn,
            device=device,
            teacher_forcing=teacher_forcing,
            grad_clip_norm=grad_clip_norm,
        )
        test_mae,test_mse,test_mape = evaluate_darnn(model, test_loader, loss_fn, device=device)

        print(f"Epoch {epoch:03d} | Train MAE: {train_mae:.4f} | Test MAE: {test_mae:.4f} | Test MSE: {test_mse:.4f} | Test MAPE: {test_mape:.2f}%")

    return model, test_mae,test_mse,test_mape

