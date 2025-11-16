# interventions/evaluate/spillover.py
import torch

@torch.no_grad()
def non_confused_indices(y_true, y_pred):
    return torch.nonzero(y_true == y_pred, as_tuple=False).view(-1)

@torch.no_grad()
def spillover_acc_drop(X, y, W_before, b_before, W_after, b_after, idx=None):
    """
    Measure accuracy change on idx (non-confused by default).
    """
    if idx is None:
        idx = non_confused_indices(
            (X @ W_before.T + b_before).argmax(1),
            (X @ W_before.T + b_before).argmax(1)
        )
    Xa, ya = X[idx], y[idx]
    acc0 = ((Xa @ W_before.T + b_before).argmax(1) == ya).float().mean().item()
    acc1 = ((Xa @ W_after.T  + b_after ).argmax(1) == ya).float().mean().item()
    return acc0 - acc1
