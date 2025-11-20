# interventions/selectors/cis.py
import torch

def class_pair_impact(W: torch.Tensor, true_idx: int, pred_idx: int):
    """
    g_j = |W_true,j - W_pred,j|
    """
    diff = (W[true_idx] - W[pred_idx]).abs()  # [D]
    return diff

def rank_by_cis(features: torch.Tensor, W: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor, topk=1):
    """
    For each sample i, build |W_t - W_p| and pick top-k concepts.
    Returns [N, topk] LongTensor.
    """
    N, D = features.shape
    out = torch.zeros((N, topk), dtype=torch.long, device=features.device)
    for i in range(N):
        g = class_pair_impact(W, int(y_true[i]), int(y_pred[i]))
        out[i] = torch.topk(g, k=topk, largest=True).indices
    return out
