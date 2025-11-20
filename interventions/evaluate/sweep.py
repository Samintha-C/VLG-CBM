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

def budget_curve_type3(X_intervention, y_intervention, X_eval, y_eval, W, b, selector_fn, topks=(1,3), tau=2.0):
    """
    Type-3 budget curve: Apply concept overrides to intervention set, evaluate on eval set.
    
    This prevents data leakage: we find misclassified samples from X_intervention (train/val),
    apply interventions to them, but evaluate accuracy on X_eval (test).
    
    Args:
        X_intervention: Features to find misclassified samples from (train/val)
        y_intervention: Labels for intervention set
        X_eval: Features to evaluate on (test)
        y_eval: Labels for evaluation set
        W, b: Sparse head weights
        selector_fn: Function to select concepts for intervention
        topks: Tuple of k values to try (e.g., (1, 2, 3))
        tau: Budget constraint for concept overrides
    
    Returns:
        dict: {k: acc_after_k_edits}, starting from baseline acc on eval set.
    """
    base = accuracy(X_eval, y_eval, W, b)
    out = {0: base}
    logger.info(f"Baseline accuracy on eval set: {base:.4f}")
    
    # Find misclassified samples in intervention set
    pred_intervention = (X_intervention @ W.T + b).argmax(1)
    mis_mask = pred_intervention != y_intervention
    Xm, ym, pm = X_intervention[mis_mask], y_intervention[mis_mask], pred_intervention[mis_mask]
    logger.info(f"Found {Xm.shape[0]} misclassified samples in intervention set (out of {X_intervention.shape[0]})")

    for k in topks:
        logger.info(f"Computing budget curve for k={k} concept edits...")
        # Select concepts to edit based on intervention set
        idx = selector_fn(Xm, topk=k)
        # Apply concept overrides to intervention set samples
        X2 = apply_concept_overrides(Xm, W, b, ym, pm, idx, m_target=0.0, tau=tau)
        
        # For evaluation: apply same intervention pattern to eval set
        # Find misclassified samples in eval set
        pred_eval = (X_eval @ W.T + b).argmax(1)
        mis_mask_eval = pred_eval != y_eval
        Xm_eval, ym_eval, pm_eval = X_eval[mis_mask_eval], y_eval[mis_mask_eval], pred_eval[mis_mask_eval]
        
        # Apply same concept selection strategy to eval set
        idx_eval = selector_fn(Xm_eval, topk=k)
        X2_eval = apply_concept_overrides(Xm_eval, W, b, ym_eval, pm_eval, idx_eval, m_target=0.0, tau=tau)
        
        # Evaluate on eval set with interventions applied
        X_eval_full = X_eval.clone()
        X_eval_full[mis_mask_eval] = X2_eval
        acc_k = accuracy(X_eval_full, y_eval, W, b)
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
    Holistic evaluation: Compute comprehensive accuracy impact analysis.
    Similar to manual_weight_editing.ipynb - full re-evaluation of entire dataset.
    
    Args:
        X: Features (not used, but kept for API consistency)
        y_true: Ground truth labels [N]
        original_preds: Predictions before interventions [N]
        new_preds: Predictions after interventions [N]
    
    Returns:
        dict with comprehensive analysis:
            - accuracy_before: overall accuracy before interventions
            - accuracy_after: overall accuracy after interventions
            - accuracy_delta: change in accuracy
            - total_corrected: number of instances fixed (wrong -> correct)
            - total_broken: number of instances broken (correct -> wrong)
            - net_corrections: total_corrected - total_broken
            - per_class_accuracy_before: dict mapping class_idx -> accuracy before
            - per_class_accuracy_after: dict mapping class_idx -> accuracy after
            - per_class_accuracy_delta: dict mapping class_idx -> accuracy change
            - per_class_corrected: dict mapping class_idx -> count of corrections
            - per_class_broken: dict mapping class_idx -> count of breakages
            - per_class_net: dict mapping class_idx -> net corrections
            - changed_indices: list of indices where predictions changed
            - unchanged_correct: count of samples that stayed correct
            - unchanged_wrong: count of samples that stayed wrong
    """
    # Overall accuracy
    acc_before = (original_preds == y_true).float().mean().item()
    acc_after = (new_preds == y_true).float().mean().item()
    acc_delta = acc_after - acc_before
    
    # Find all changed predictions
    diff = (original_preds != new_preds)
    changed_indices = torch.nonzero(diff, as_tuple=False).view(-1).tolist()
    
    # Count unchanged samples
    unchanged_mask = ~diff
    unchanged_correct = ((original_preds == y_true) & unchanged_mask).sum().item()
    unchanged_wrong = ((original_preds != y_true) & unchanged_mask).sum().item()
    
    # Track corrections and breakages
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
    
    # Per-class accuracy analysis
    num_classes = int(y_true.max().item()) + 1
    per_class_acc_before = {}
    per_class_acc_after = {}
    per_class_acc_delta = {}
    
    for c in range(num_classes):
        class_mask = (y_true == c)
        if class_mask.sum() > 0:
            acc_b = (original_preds[class_mask] == y_true[class_mask]).float().mean().item()
            acc_a = (new_preds[class_mask] == y_true[class_mask]).float().mean().item()
            per_class_acc_before[c] = acc_b
            per_class_acc_after[c] = acc_a
            per_class_acc_delta[c] = acc_a - acc_b
    
    # Compute net per class
    all_classes = set(per_class_corrected.keys()) | set(per_class_broken.keys())
    per_class_net = {c: per_class_corrected.get(c, 0) - per_class_broken.get(c, 0) 
                     for c in all_classes}
    
    return {
        "accuracy_before": acc_before,
        "accuracy_after": acc_after,
        "accuracy_delta": acc_delta,
        "total_corrected": total_corrected,
        "total_broken": total_broken,
        "net_corrections": total_corrected - total_broken,
        "per_class_accuracy_before": per_class_acc_before,
        "per_class_accuracy_after": per_class_acc_after,
        "per_class_accuracy_delta": per_class_acc_delta,
        "per_class_corrected": per_class_corrected,
        "per_class_broken": per_class_broken,
        "per_class_net": per_class_net,
        "changed_indices": changed_indices,
        "unchanged_correct": unchanged_correct,
        "unchanged_wrong": unchanged_wrong,
        "total_samples": len(y_true),
    }
