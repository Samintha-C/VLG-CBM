# interventions/evaluate/sweep.py
import torch
from ..editors.concepts import apply_concept_overrides
from ..editors.weights import nudge_final_layer, margin

@torch.no_grad()
def accuracy(X, y, W, b, batch=4096):
    acc = 0.0
    for i in range(0, X.shape[0], batch):
        logits = X[i:i+batch] @ W.T + b
        acc += (logits.argmax(1) == y[i:i+batch]).float().sum().item()
    return acc / X.shape[0]

def budget_curve_type3(X, y, W, b, selector_fn, topks=(1,3), tau=2.0):
    """
    Return dict: {k: acc_after_k_edits}, starting from baseline acc.
    """
    base = accuracy(X, y, W, b)
    out = {0: base}
    # only edit misclassified samples (speed)
    pred = (X @ W.T + b).argmax(1)
    mis_mask = pred != y
    Xm, ym, pm = X[mis_mask], y[mis_mask], pred[mis_mask]

    for k in topks:
        idx = selector_fn(Xm, topk=k)               # [M, k]
        X2 = apply_concept_overrides(Xm, W, b, ym, pm, idx, m_target=0.0, tau=tau)
        X_full = X.clone()
        X_full[mis_mask] = X2
        out[k] = accuracy(X_full, y, W, b)
    return out

def weight_nudge_eval(X_train, y_train, X_val, y_val, W, b,
                      chosen_indices_fn, tau=1e-2, sample_limit=1000):
    """
    Try nudges on a subset of misclassified train samples; accept if val acc doesn't drop.
    Returns possibly-updated W,b and a small log.
    """
    log = []
    logits_train = X_train @ W.T + b
    pred_train = logits_train.argmax(1)
    mis_idx = torch.nonzero(pred_train != y_train, as_tuple=False).view(-1)[:sample_limit]

    W2, b2 = W.clone(), b.clone()
    base_val = accuracy(X_val, y_val, W2, b2)
    for i in mis_idx.tolist():
        t, p = int(y_train[i]), int(pred_train[i])
        js = chosen_indices_fn(X_train[i:i+1], topk=1)[0].tolist()
        W_try, b_try = nudge_final_layer(W2, b2, X_train[i], t, p, js, tau=tau)
        new_val = accuracy(X_val, y_val, W_try, b_try)
        if new_val + 1e-6 >= base_val:  # accept if no val harm
            W2, b2 = W_try, b_try
            base_val = new_val
            log.append({"i": i, "t": t, "p": p, "js": js})
    return W2, b2, log
