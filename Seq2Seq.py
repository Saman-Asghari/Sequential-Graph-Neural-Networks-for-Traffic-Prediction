
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def prepare_seq2seq_data(X: torch.Tensor, Y: torch.Tensor):
    """
    X: (B, T_in, N, 1)
    Y: (B, T_out, N, 1)

    returns:
      X_seq: (B*N, T_in, 1)
      Y_seq: (B*N, T_out, 1)
    """
    if X.dim() == 3:
        X = X.unsqueeze(-1)
    if Y.dim() == 3:
        Y = Y.unsqueeze(-1)

    assert X.dim() == 4 and Y.dim() == 4, f"Expected 4D tensors. Got X={X.shape}, Y={Y.shape}"
    B, T_in, N, F = X.shape
    assert F == 1, f"Expected feature dim=1, got {F}"
    T_out = Y.shape[1]

    X = X.squeeze(-1)
    Y = Y.squeeze(-1)

    X = X.permute(0, 2, 1).contiguous()
    Y = Y.permute(0, 2, 1).contiguous()

    # (B*N, T_in, 1) and (B*N, T_out, 1)
    X_seq = X.view(B * N, T_in).unsqueeze(-1)
    Y_seq = Y.view(B * N, T_out).unsqueeze(-1)

    return X_seq, Y_seq


class Seq2SeqLSTM(nn.Module):
   
    def __init__(self, input_size=1, hidden_size=10, num_layers=3, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Note: dropout in nn.LSTM is applied between layers when num_layers > 1
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.decoder = nn.LSTM(
            input_size=1,  # we feed previous y (scalar) each step
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(self, x_enc, y_true=None, out_len=12, teacher_forcing=0.5):
       
        B = x_enc.size(0)

        # Encode
        _, (h, c) = self.encoder(x_enc)  # h,c: (num_layers, B, hidden)

        # Decoder initial input: last encoder input value (common baseline choice)
        dec_in = x_enc[:, -1:, :]  # (B, 1, 1)

        preds = []
        for t in range(out_len):
            dec_out, (h, c) = self.decoder(dec_in, (h, c))  # dec_out: (B, 1, hidden)
            y_step = self.out_proj(dec_out)                 # (B, 1, 1)
            preds.append(y_step)

            if y_true is not None and np.random.rand() < teacher_forcing:
                # teacher forcing: feed ground-truth previous step
                dec_in = y_true[:, t:t+1, :]
            else:
                # feed model's own prediction
                dec_in = y_step

        y_pred = torch.cat(preds, dim=1)  # (B, T_out, 1)
        return y_pred

def train_one_epoch(model, train_loader, optimizer, loss_fn, device, teacher_forcing=0.5):
    model.train()
    total_loss = 0.0

    for Xb, Yb in train_loader:
        Xb = Xb.to(device)
        Yb = Yb.to(device)

        X_seq, Y_seq = prepare_seq2seq_data(Xb, Yb)

        optimizer.zero_grad(set_to_none=True)
        Y_pred = model(
            X_seq,
            y_true=Y_seq,
            out_len=Y_seq.size(1),
            teacher_forcing=teacher_forcing,
        )
        loss = loss_fn(Y_pred, Y_seq)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(train_loader))
@torch.no_grad()
def evaluate(model, test_loader, loss_fn, device, eps=1e-5):
   
    model.eval()
    total_mae = 0.0
    total_mse = 0.0
    total_mape = 0.0

    mse_fn = torch.nn.MSELoss(reduction="mean")

    for Xb, Yb in test_loader:
        Xb = Xb.to(device)
        Yb = Yb.to(device)

        X_seq, Y_seq = prepare_seq2seq_data(Xb, Yb)

        Y_pred = model(
            X_seq,
            y_true=None,
            out_len=Y_seq.size(1),
            teacher_forcing=0.0,
        )

        # MAE
        total_mae += loss_fn(Y_pred, Y_seq).item()

        # MSE
        total_mse += mse_fn(Y_pred, Y_seq).item()

        # MAPE (safe)
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


def train(model,train_loader, test_loader,teacher_forcing,device, epochs=50, lr=1e-3,loss_fn = nn.L1Loss()):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        train_mae = train_one_epoch(model, train_loader, optimizer, loss_fn,device ,teacher_forcing=teacher_forcing)
        test_mae,test_mse,test_mape = evaluate(model, test_loader, loss_fn,device)
        print(f"Epoch {epoch:03d} | Train MAE: {train_mae:.4f} | Test MAE: {test_mae:.4f} | Test MSE: {test_mse:.4f} | Test MAPE: {test_mape:.2f}%")
    return model,test_mae,test_mse,test_mape