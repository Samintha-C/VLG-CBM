# interventions/evaluate/sweep.py
import torch
from loguru import logger
from ..editors.concepts import apply_concept_overrides
from ..editors.weights import nudge_final_layer, margin

@torch.no_grad()
def accuracy(X, y, W, b, batch=4096):
    acc = 0.0
    for i in range(0, X.shape[0], batch):
        logits = X[i:i+batch] @ W.T + b
        acc += (logits.argmax(1) == y[i:i+batch]).float().sum().item()
    return acc / X.shape[0]

@torch.no_grad()
def get_predictions(X, W, b, batch=4096):
    """Get predictions for all samples in X. Returns tensor of class indices."""
    preds = []
    for i in range(0, X.shape[0], batch):
        logits = X[i:i+batch] @ W.T + b
        preds.append(logits.argmax(1))
    return torch.cat(preds, dim=0)

def budget_curve_type3(X, y, W, b, selector_fn, topks=(1,3), tau=2.0):
    """
    Return dict: {k: acc_after_k_edits}, starting from baseline acc.
    """
    base = accuracy(X, y, W, b)
    out = {0: base}
    logger.info(f"Baseline accuracy: {base:.4f}")
    
    pred = (X @ W.T + b).argmax(1)
    mis_mask = pred != y
    Xm, ym, pm = X[mis_mask], y[mis_mask], pred[mis_mask]
    logger.info(f"Found {Xm.shape[0]} misclassified samples out of {X.shape[0]}")

    for k in topks:
        logger.info(f"Computing budget curve for k={k} concept edits...")
        idx = selector_fn(Xm, topk=k)
        X2 = apply_concept_overrides(Xm, W, b, ym, pm, idx, m_target=0.0, tau=tau)
        X_full = X.clone()
        X_full[mis_mask] = X2
        acc_k = accuracy(X_full, y, W, b)
        out[k] = acc_k
        logger.info(f"  k={k}: accuracy = {acc_k:.4f} (delta: {acc_k-base:+.4f})")
    return out

def weight_nudge_eval(X_train, y_train, X_val, y_val, W, b,
                      chosen_indices_fn, tau=1e-2, sample_limit=1000):
    """
    Try nudges on a subset of misclassified train samples; accept if val acc doesn't drop.
    Returns possibly-updated W,b and a small log.
    """
    log = []
    logger.info("Finding misclassified train samples...")
    logits_train = X_train @ W.T + b
    pred_train = logits_train.argmax(1)
    mis_idx = torch.nonzero(pred_train != y_train, as_tuple=False).view(-1)[:sample_limit]
    logger.info(f"Found {len(mis_idx)} misclassified samples (limited to {sample_limit})")

    W2, b2 = W.clone(), b.clone()
    base_val = accuracy(X_val, y_val, W2, b2)
    logger.info(f"Baseline val accuracy: {base_val:.4f}")
    logger.info(f"Processing {len(mis_idx)} samples for weight nudges...")

    accepted = 0
    for idx, i in enumerate(mis_idx.tolist()):
        if (idx + 1) % 100 == 0:
            logger.info(f"  Processed {idx+1}/{len(mis_idx)} samples, accepted {accepted} edits")
        t, p = int(y_train[i]), int(pred_train[i])
        js = chosen_indices_fn(X_train[i:i+1], topk=1)[0].tolist()
        W_try, b_try = nudge_final_layer(W2, b2, X_train[i], t, p, js, tau=tau)
        new_val = accuracy(X_val, y_val, W_try, b_try)
        if new_val + 1e-6 >= base_val:
            W2, b2 = W_try, b_try
            base_val = new_val
            accepted += 1
            log.append({"i": i, "t": t, "p": p, "js": js})
    
    logger.info(f"Accepted {accepted} weight nudges, final val acc: {base_val:.4f}")
    return W2, b2, log

def compute_net_corrections(X, y_true, original_preds, new_preds):
    """
    Compute net corrections: instances corrected - instances broken.
    
    Args:
        X: Features (not used, but kept for API consistency)
        y_true: Ground truth labels [N]
        original_preds: Predictions before interventions [N]
        new_preds: Predictions after interventions [N]
    
    Returns:
        dict with:
            - total_corrected: number of instances fixed (wrong -> correct)
            - total_broken: number of instances broken (correct -> wrong)
            - net_corrections: total_corrected - total_broken
            - per_class_corrected: dict mapping class_idx -> count of corrections
            - per_class_broken: dict mapping class_idx -> count of breakages
            - per_class_net: dict mapping class_idx -> net corrections
            - changed_indices: list of indices where predictions changed
    """
    diff = (original_preds != new_preds)
    changed_indices = torch.nonzero(diff, as_tuple=False).view(-1).tolist()
    
    total_corrected = 0
    total_broken = 0
    per_class_corrected = {}
    per_class_broken = {}
    
    for idx in changed_indices:
        gt = int(y_true[idx])
        orig_pred = int(original_preds[idx])
        new_pred = int(new_preds[idx])
        
        if gt == orig_pred:
            # Was correct, now wrong (broken)
            total_broken += 1
            per_class_broken[gt] = per_class_broken.get(gt, 0) + 1
        elif gt == new_pred:
            # Was wrong, now correct (corrected)
            total_corrected += 1
            per_class_corrected[gt] = per_class_corrected.get(gt, 0) + 1
    
    # Compute net per class
    all_classes = set(per_class_corrected.keys()) | set(per_class_broken.keys())
    per_class_net = {c: per_class_corrected.get(c, 0) - per_class_broken.get(c, 0) 
                     for c in all_classes}
    
    return {
        "total_corrected": total_corrected,
        "total_broken": total_broken,
        "net_corrections": total_corrected - total_broken,
        "per_class_corrected": per_class_corrected,
        "per_class_broken": per_class_broken,
        "per_class_net": per_class_net,
        "changed_indices": changed_indices,
    }
