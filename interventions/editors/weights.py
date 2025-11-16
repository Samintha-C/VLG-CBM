# interventions/editors/weights.py
import torch

def nudge_final_layer(W: torch.Tensor, b: torch.Tensor,
                      x_row: torch.Tensor, true_idx: int, pred_idx: int,
                      selected_js: list[int], tau: float = 1e-2):
    """
    Tiny, local L2-minimal nudge on W_t[S], W_p[S].
    """
    S = torch.tensor(selected_js, dtype=torch.long)
    xS = x_row[S]                     # [L]
    need = max(0.0, 0.0 - float((W[true_idx]-W[pred_idx]) @ x_row + (b[true_idx]-b[pred_idx])))
    denom = float(xS @ xS + 1e-8)
    alpha = 0.0 if need == 0.0 else need / denom
    # enforce budget: scale alpha so ||ΔW|| <= tau (approx via L2 on the segment)
    # ΔW rows each get 0.5*alpha*xS on |S|
    delta_norm = (0.5*alpha) * float(xS.norm())
    if delta_norm > tau and delta_norm > 0:
        alpha = alpha * (tau / delta_norm)

    d = 0.5 * alpha * xS              # [L]
    W = W.clone()
    W[true_idx, S] += d
    W[pred_idx, S] -= d
    return W, b  # biases unchanged

def margin(W, b, x, t, p):
    return float((W[t]-W[p]) @ x + (b[t]-b[p]))
