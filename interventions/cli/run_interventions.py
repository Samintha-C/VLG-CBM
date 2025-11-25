# interventions/cli/run_interventions.py
import argparse, torch
from loguru import logger
from ..adapters.vlgcbm import VLGCbmRun, get_loader, load_sparse_head
from ..selectors.entropy import rank_uncertain_concepts
from ..selectors.confusion import top_confusions, bucket_indices
from ..evaluate.sweep import budget_curve_type3, weight_nudge_eval, accuracy, get_predictions, compute_net_corrections
from ..evaluate.report import stamp_dir, save_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", choices=["vlg"], default="vlg")
    ap.add_argument("--load_path", required=True)
    ap.add_argument("--nec", type=int, required=True)
    ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--tau_concept", type=float, default=2.0)
    ap.add_argument("--tau_weight", type=float, default=1e-2)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Load path: {args.load_path}, NEC: {args.nec}, Budget: {args.budget}")

    outdir = stamp_dir()
    logger.info(f"Output directory: {outdir}")

    logger.info("Loading sparse head...")
    run = VLGCbmRun(load_path=args.load_path, nec=args.nec)
    W, b, C = load_sparse_head(run, device=device)
    logger.info(f"Loaded W shape: {W.shape}, b shape: {b.shape}, num_classes: {C}")

    logger.info("Loading data splits...")
    logger.info("="*70)
    logger.info("DATA SPLIT STRATEGY:")
    logger.info("  Train: Training data (not used for interventions) - keeping on CPU to save GPU memory")
    logger.info("  Val: Find mistakes, apply interventions (concept recalibration) - on GPU")
    logger.info("  Test: Pure evaluation (unseen, not used in any intervention decisions) - on GPU")
    logger.info("="*70)
    
    # For large datasets like ImageNet, keep train on CPU (we don't use it for interventions)
    tr = get_loader(run, "train", batch_size=4096, device="cpu")
    va = get_loader(run, "val", batch_size=4096, device=device)
    
    # Try to load test set, but if it doesn't exist, we'll split val set
    try:
        te = get_loader(run, "test", batch_size=4096, device=device)
        has_test = True
        logger.info("Found separate test set")
    except FileNotFoundError:
        logger.info("Test set not found - will split validation set into intervention and test portions")
        has_test = False

    logger.info("Materializing tensors...")
    if hasattr(tr.dataset, "tensors"):
        Xtr, ytr = tr.dataset.tensors[0], tr.dataset.tensors[1]
        Xva, yva = va.dataset.tensors[0], va.dataset.tensors[1]
        if has_test:
            Xte, yte = te.dataset.tensors[0], te.dataset.tensors[1]
    else:
        logger.info("Concatenating batches...")
        Xtr = torch.cat([X for X,_ in tr], 0)
        ytr = torch.cat([y for _,y in tr], 0)
        Xva = torch.cat([X for X,_ in va], 0)
        yva = torch.cat([y for _,y in va], 0)
        if has_test:
            Xte = torch.cat([X for X,_ in te], 0)
            yte = torch.cat([y for _,y in te], 0)

    # If test set doesn't exist, split val set into intervention and test
    if not has_test:
        logger.info("="*70)
        logger.info("SPLITTING VALIDATION SET:")
        logger.info("  Original val set will be split into:")
        logger.info("    - Val (intervention): For finding mistakes and applying interventions")
        logger.info("    - Test: Unseen evaluation set (split from val)")
        logger.info("="*70)
        
        # Split val set: 50% for interventions, 50% for test
        n_val = len(Xva)
        n_int = n_val // 2
        n_test = n_val - n_int
        
        # Random permutation for splitting (with fixed seed for reproducibility)
        torch.manual_seed(42)
        indices = torch.randperm(n_val, device=device)
        int_indices = indices[:n_int]
        test_indices = indices[n_int:]
        
        # Indexing preserves device, so Xva_int and Xte will be on the same device as Xva
        Xva_int = Xva[int_indices]
        yva_int = yva[int_indices]
        Xte = Xva[test_indices]
        yte = yva[test_indices]
        
        # Replace val with intervention portion
        Xva, yva = Xva_int, yva_int
        
        logger.info(f"Split validation set ({n_val} samples):")
        logger.info(f"  Val (intervention): {len(Xva)} samples")
        logger.info(f"  Test (from val split): {len(Xte)} samples")
    else:
        logger.info("Using separate test set (not split from val)")
    
    logger.info(f"Train: {Xtr.shape[0]} samples (on CPU, not used for interventions)")
    logger.info(f"Val: {Xva.shape[0]} samples (for interventions, on {device})")
    logger.info(f"Test: {Xte.shape[0]} samples (unseen, for evaluation only, on {device})")
    # Train stays on CPU (we don't use it), val and test are already on device from get_loader
    # Just ensure val and test are on device (they should be already)
    if Xva.device != torch.device(device):
        logger.info("Moving val to device...")
        Xva, yva = Xva.to(device), yva.to(device)
    if Xte.device != torch.device(device):
        logger.info("Moving test to device...")
        Xte, yte = Xte.to(device), yte.to(device)

    logger.info("Computing baseline accuracy and recording original predictions...")
    base_val = accuracy(Xva, yva, W, b)
    base_test = accuracy(Xte, yte, W, b)
    logger.info(f"Baseline val acc (NEC={args.nec}): {base_val:.4f}")
    logger.info(f"Baseline test acc (NEC={args.nec}): {base_test:.4f}")

    # Record original predictions for net correction analysis on TEST set (truly unseen)
    logger.info("Recording baseline predictions on TEST set (unseen data)...")
    original_preds_test = get_predictions(Xte, W, b)
    logger.info(f"Recorded predictions for {len(original_preds_test)} test samples")

    logger.info("Starting Type-3 budget curve (concept overrides)...")
    logger.info("  Pattern: Intervene on val set, evaluate on same set (holistic analysis)")
    logger.info("  This matches manual_weight_editing.ipynb: see holistic impact of interventions")
    selector = lambda X, topk: rank_uncertain_concepts(X, topk=topk, T=2.0)
    # Match manual experiment: intervene and evaluate on same set (val set for analysis)
    curve = budget_curve_type3(Xva, yva, W, b, selector_fn=selector, 
                               topks=tuple(range(1, args.budget+1)), tau=args.tau_concept)
    logger.info(f"Type-3 (concept overrides) acc vs budget: {curve}")

    logger.info("Starting Type-4 weight nudges...")
    logger.info("  Workflow: Find mistakes in Val → Apply interventions → Accept if val acc doesn't drop")
    logger.info("  Then evaluate on Test set (unseen) to check generalization")
    
    # Find mistakes in val set and apply interventions
    # Accept interventions only if val accuracy doesn't drop (prevents overfitting)
    W2, b2, log = weight_nudge_eval(Xva, yva, Xva, yva, W, b, 
                                     chosen_indices_fn=selector, tau=args.tau_weight, sample_limit=1000)
    logger.info(f"Type-4: Processed {len(log)} accepted edits out of 1000 attempts")
    
    # Check val accuracy after interventions
    base_val_after = accuracy(Xva, yva, W2, b2)
    logger.info(f"Val acc: {base_val:.4f} → {base_val_after:.4f} (delta: {base_val_after-base_val:+.4f})")
    
    # Final evaluation on test set (unseen)
    base_test_after = accuracy(Xte, yte, W2, b2)
    logger.info(f"Test acc (unseen): {base_test:.4f} → {base_test_after:.4f} (delta: {base_test_after-base_test:+.4f})  accepted_edits={len(log)}")

    # Holistic re-evaluation: Full accuracy impact analysis on TEST set (truly unseen)
    logger.info("Performing holistic re-evaluation on TEST set (unseen data)...")
    new_preds_test = get_predictions(Xte, W2, b2)
    net_corrections = compute_net_corrections(Xte, yte, original_preds_test, new_preds_test)
    
    logger.info("="*70)
    logger.info("HOLISTIC ACCURACY IMPACT ANALYSIS (TEST SET - UNSEEN DATA)")
    logger.info("="*70)
    logger.info(f"Overall Accuracy: {net_corrections['accuracy_before']:.4f} → {net_corrections['accuracy_after']:.4f} "
                f"(delta: {net_corrections['accuracy_delta']:+.4f})")
    logger.info(f"Total samples: {net_corrections['total_samples']}")
    logger.info(f"  Corrected: {net_corrections['total_corrected']} (wrong → correct)")
    logger.info(f"  Broken: {net_corrections['total_broken']} (correct → wrong)")
    logger.info(f"  Net corrections: {net_corrections['net_corrections']:+d}")
    logger.info(f"  Unchanged (correct): {net_corrections['unchanged_correct']}")
    logger.info(f"  Unchanged (wrong): {net_corrections['unchanged_wrong']}")
    logger.info("="*70)

    logger.info("Saving results...")
    save_json({
        "baseline_val_acc": base_val,
        "baseline_test_acc": base_test,
        "T3_curve": curve, 
        "T4_val_acc": base_val_after,
        "T4_test_acc": base_test_after,
        "T4_log": log,
        "net_corrections": net_corrections,
        "nec": args.nec, 
        "load_path": args.load_path
    }, outdir, "summary")
    logger.info(f"Results saved to {outdir}/summary.json")

if __name__ == "__main__":
    main()
