# interventions/cli/run_interventions.py
import argparse, torch
from loguru import logger
from ..adapters.vlgcbm import VLGCbmRun, get_loader, load_sparse_head, split_validation_set
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
    ap.add_argument("--val_intervention_ratio", type=float, default=0.5, 
                    help="Fraction of validation set to use for interventions (rest for evaluation)")
    args = ap.parse_args()

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Load path: {args.load_path}, NEC: {args.nec}, Budget: {args.budget}")

    outdir = stamp_dir()
    logger.info(f"Output directory: {outdir}")

    logger.info("Loading sparse head...")
    run = VLGCbmRun(load_path=args.load_path, nec=args.nec)
    W, b, C = load_sparse_head(run)
    logger.info(f"Loaded W shape: {W.shape}, b shape: {b.shape}, num_classes: {C}")
    W, b = W.to(device), b.to(device)

    logger.info("Loading data splits...")
    logger.info("="*70)
    logger.info("DATA SPLIT STRATEGY:")
    logger.info("  Train: Training data (not used for interventions)")
    logger.info("  Val (intervention): Find mistakes, apply interventions")
    logger.info("  Val (eval): Evaluate intervened model (generalization check)")
    logger.info("  Test: Final evaluation (if available)")
    logger.info("="*70)
    
    tr = get_loader(run, "train", batch_size=4096)
    
    # Split validation set into intervention and evaluation portions
    X_val_int, y_val_int, X_val_eval, y_val_eval = split_validation_set(
        run, intervention_ratio=args.val_intervention_ratio, seed=42
    )
    
    # Try to load test set, but it might not exist
    try:
        te = get_loader(run, "test", batch_size=4096)
        has_test = True
    except FileNotFoundError:
        logger.warning("Test set not found, will use val_eval as final evaluation set")
        has_test = False

    logger.info("Materializing tensors...")
    if hasattr(tr.dataset, "tensors"):
        Xtr, ytr = tr.dataset.tensors[0], tr.dataset.tensors[1]
        if has_test:
            Xte, yte = te.dataset.tensors[0], te.dataset.tensors[1]
    else:
        logger.info("Concatenating batches...")
        Xtr = torch.cat([X for X,_ in tr], 0)
        ytr = torch.cat([y for _,y in tr], 0)
        if has_test:
            Xte = torch.cat([X for X,_ in te], 0)
            yte = torch.cat([y for _,y in te], 0)
    
    # Use val_eval as test if test set doesn't exist
    if not has_test:
        Xte, yte = X_val_eval, y_val_eval
        logger.info("Using val_eval as final evaluation set (test set not available)")
    
    logger.info(f"Train: {Xtr.shape[0]} samples")
    logger.info(f"Val (intervention): {X_val_int.shape[0]} samples")
    logger.info(f"Val (eval): {X_val_eval.shape[0]} samples")
    logger.info(f"Test: {Xte.shape[0]} samples")
    logger.info("Moving tensors to device...")
    Xtr, ytr = Xtr.to(device), ytr.to(device)
    X_val_int, y_val_int = X_val_int.to(device), y_val_int.to(device)
    X_val_eval, y_val_eval = X_val_eval.to(device), y_val_eval.to(device)
    Xte, yte = Xte.to(device), yte.to(device)

    logger.info("Computing baseline accuracy and recording original predictions...")
    # Compute baselines on both val_eval and test
    base_val_eval = accuracy(X_val_eval, y_val_eval, W, b)
    base_test = accuracy(Xte, yte, W, b)
    logger.info(f"Baseline val (eval) acc (NEC={args.nec}): {base_val_eval:.4f}")
    logger.info(f"Baseline test acc (NEC={args.nec}): {base_test:.4f}")
    
    # Record original predictions for net correction analysis on val_eval
    logger.info("Recording baseline predictions on val (eval) set...")
    original_preds = get_predictions(X_val_eval, W, b)
    logger.info(f"Recorded predictions for {len(original_preds)} val (eval) samples")

    logger.info("Starting Type-3 budget curve (concept overrides)...")
    logger.info("  Pattern: Intervene on val (eval) set, evaluate on same set (holistic analysis)")
    logger.info("  This matches manual_weight_editing.ipynb: see holistic impact of interventions")
    selector = lambda X, topk: rank_uncertain_concepts(X, topk=topk, T=2.0)
    # Match manual experiment: intervene and evaluate on same set (val_eval for analysis)
    curve = budget_curve_type3(X_val_eval, y_val_eval, W, b, selector_fn=selector, 
                               topks=tuple(range(1, args.budget+1)), tau=args.tau_concept)
    logger.info(f"Type-3 (concept overrides) acc vs budget: {curve}")

    logger.info("Starting Type-4 weight nudges...")
    logger.info("  Workflow: Find mistakes in Val (intervention) → Apply interventions → Evaluate on Val (eval) → Test")
    logger.info("  Val (intervention) serves as intervention test ground")
    logger.info("  Val (eval) checks if interventions generalize")
    logger.info("  Test provides final evaluation")
    
    # Find mistakes in val_intervention set and apply interventions
    # Use val_eval to check if interventions don't degrade performance
    W2, b2, log = weight_nudge_eval(X_val_int, y_val_int, X_val_eval, y_val_eval, W, b, 
                                     chosen_indices_fn=selector, tau=args.tau_weight, sample_limit=1000)
    logger.info(f"Type-4: Processed {len(log)} accepted edits out of 1000 attempts")
    
    # Evaluate on val_eval (generalization check)
    base_val_eval_after = accuracy(X_val_eval, y_val_eval, W2, b2)
    logger.info(f"Val (eval) acc: {base_val_eval:.4f} → {base_val_eval_after:.4f} (delta: {base_val_eval_after-base_val_eval:+.4f})")
    
    # Final evaluation on test set
    base_test_after = accuracy(Xte, yte, W2, b2)
    logger.info(f"Test acc: {base_test:.4f} → {base_test_after:.4f} (delta: {base_test_after-base_test:+.4f})  accepted_edits={len(log)}")

    # Holistic re-evaluation: Full accuracy impact analysis on val_eval
    logger.info("Performing holistic re-evaluation on val (eval) set...")
    new_preds = get_predictions(X_val_eval, W2, b2)
    net_corrections = compute_net_corrections(X_val_eval, y_val_eval, original_preds, new_preds)
    
    logger.info("="*70)
    logger.info("HOLISTIC ACCURACY IMPACT ANALYSIS")
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
        "baseline_val_eval_acc": base_val_eval,
        "baseline_test_acc": base_test,
        "T3_curve": curve, 
        "T4_val_eval_acc": base_val_eval_after,
        "T4_test_acc": base_test_after,
        "T4_log": log,
        "net_corrections": net_corrections,
        "nec": args.nec, 
        "load_path": args.load_path,
        "val_intervention_ratio": args.val_intervention_ratio
    }, outdir, "summary")
    logger.info(f"Results saved to {outdir}/summary.json")

if __name__ == "__main__":
    main()
