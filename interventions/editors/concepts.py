# interventions/editors/concepts.py
import torch

def minimal_single_edit_delta(x_row: torch.Tensor, W: torch.Tensor, b: torch.Tensor, true_idx: int, pred_idx: int,
                              concept_j: int, m_target: float = 0.0, tau: float = 2.0):
    """
    Compute minimal delta on x_j to reach margin >= m_target.
    Returns delta scalar.
    """
    margin = (W[true_idx] - W[pred_idx]) @ x_row + (b[true_idx] - b[pred_idx])
    need = max(0.0, m_target - float(margin))
    denom = float(abs(W[true_idx, concept_j] - W[pred_idx, concept_j]) + 1e-8)
    sign = 1.0 if (W[true_idx, concept_j] - W[pred_idx, concept_j]) >= 0 else -1.0
    delta = min(need/denom, tau) * sign
    return delta

def apply_concept_overrides(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor,
                            y_true: torch.Tensor, y_pred: torch.Tensor,
                            chosen_indices: torch.Tensor,  # [N, L] concepts per sample
                            m_target: float = 0.0, tau: float = 2.0,
                            clip_bounds: tuple[float,float] | None = None,
                            return_edits: bool = False):
    """
    Returns a *copy* X' with per-sample small overrides on chosen concept indices.
    If return_edits=True, also returns a list of edit records.
    """
    X2 = X.clone()
    low, high = clip_bounds if clip_bounds is not None else (-float("inf"), float("inf"))
    edits = [] if return_edits else None
    
    for i in range(X.shape[0]):
        t, p = int(y_true[i]), int(y_pred[i])
        sample_edits = []
        for j in chosen_indices[i].tolist():
            old_val = X2[i, j].item()
            d = minimal_single_edit_delta(X2[i], W, b, t, p, j, m_target=m_target, tau=tau)
            X2[i, j] = X2[i, j].add_(d).clamp_(low, high)
            new_val = X2[i, j].item()
            if return_edits:
                sample_edits.append({
                    "concept_idx": int(j),
                    "delta": float(d),
                    "old_activation": old_val,
                    "new_activation": new_val,
                    "weight_diff": float(W[t, j] - W[p, j])
                })
        if return_edits and sample_edits:
            edits.append({
                "sample_idx": int(i),
                "true_class": int(t),
                "pred_class": int(p),
                "concept_edits": sample_edits
            })
    
    if return_edits:
        return X2, edits
    return X2
