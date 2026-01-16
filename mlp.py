import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class MLPBaseline(nn.Module):
    def __init__(self, input_len: int, output_len: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_len, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def prepare_mlp_data(X: torch.Tensor, Y: torch.Tensor):
    """
    Convert:
      X: (B, T_in, N, 1)  -> (B*N, T_in)
      Y: (B, T_out, N, 1) -> (B*N, T_out)
    """
    # Robustify in case last dim missing
    if X.dim() == 3:
        X = X.unsqueeze(-1)
    if Y.dim() == 3:
        Y = Y.unsqueeze(-1)

    assert X.dim() == 4 and Y.dim() == 4, f"Expected 4D tensors. Got X={X.shape}, Y={Y.shape}"

    B, T_in, N, F = X.shape
    assert F == 1, f"Expected feature dim=1 for MLP baseline, got {F}"
    T_out = Y.shape[1]

    # (B, T_in, N)
    X = X.squeeze(-1)
    Y = Y.squeeze(-1)

    # (B, N, T_in) and (B, N, T_out)
    X = X.permute(0, 2, 1).contiguous()
    Y = Y.permute(0, 2, 1).contiguous()

    # (B*N, T_in) and (B*N, T_out)
    X = X.view(B * N, T_in)
    Y = Y.view(B * N, T_out)

    return X, Y


# -------------------------
# Train / Eval helpers
# -------------------------
def train_one_epoch(model, train_loader, optimizer, loss_fn,device):
    model.train()
    total_loss = 0.0

    for X_batch, Y_batch in train_loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        X_mlp, Y_mlp = prepare_mlp_data(X_batch, Y_batch)

        optimizer.zero_grad(set_to_none=True)
        pred = model(X_mlp)
        loss = loss_fn(pred, Y_mlp)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(train_loader))


@torch.no_grad()
def evaluate(model, test_loader, loss_fn, device):
   
    model.eval()
    total_mae = 0.0
    total_mse = 0.0
    total_mape = 0.0
    eps=1e-5  

    mse_fn = torch.nn.MSELoss(reduction="mean")

    for X_batch, Y_batch in test_loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        X_mlp, Y_mlp = prepare_mlp_data(X_batch, Y_batch)
        pred = model(X_mlp)

        # MAE
        total_mae += loss_fn(pred, Y_mlp).item()

        # MSE
        total_mse += mse_fn(pred, Y_mlp).item()

        # MAPE (safe)
        batch_mape = torch.mean(
            torch.abs((Y_mlp - pred) / (Y_mlp + eps))
        ) * 100.0
        total_mape += batch_mape.item()

    denom = max(1, len(test_loader))
    return (
        total_mae / denom,
        total_mse / denom,
        total_mape / denom,
    )

@torch.no_grad()
def predict_one_batch(model, test_loader,device):
    model.eval()
    X_batch, Y_batch = next(iter(test_loader))
    X_batch = X_batch.to(device)
    Y_batch = Y_batch.to(device)

    X_mlp, Y_true_flat = prepare_mlp_data(X_batch, Y_batch)
    Y_pred_flat = model(X_mlp)  # (B*N, T_out)

    # Convert back to (B, T_out, N, 1)
    B, T_out, N, _ = Y_batch.shape
    Y_pred_full = Y_pred_flat.view(B, N, T_out).permute(0, 2, 1).unsqueeze(-1).contiguous()

    return X_batch, Y_batch, Y_pred_full, Y_pred_flat, Y_true_flat


def train(train_loader,test_loader,epochs,learning_rate,device,INPUT_SEQ_LEN=24,OUTPUT_SEQ_LEN=12,loss_fn = nn.L1Loss()):
    assert epochs > 0 

    model = MLPBaseline(INPUT_SEQ_LEN, OUTPUT_SEQ_LEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # loss_fn = nn.L1Loss()  # MAE
    print(1)
    for epoch in range(1, epochs + 1):
        train_mae = train_one_epoch(model, train_loader, optimizer, loss_fn,device)
        test_mae, test_mse, test_mape = evaluate(model, test_loader, loss_fn,device)
        print(f"Epoch {epoch:03d} | Train MAE: {train_mae:.4f} | Test MAE: {test_mae:.4f} | Test MSE: {test_mse:.4f} | Test MAPE: {test_mape:.2f}%")
    
    return model,test_mae,test_mse,test_mape





# show_prediction_with_history(
#     mlp_model,
#     test_loader,
#     device=DEVICE,
#     sample_id=300,
#     node_id=90,
#     prepare_fn=mlp.prepare_mlp_data,
#     title="MLP (Past + Future)",
# )