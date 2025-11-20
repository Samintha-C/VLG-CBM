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
    W, b, C = load_sparse_head(run)
    logger.info(f"Loaded W shape: {W.shape}, b shape: {b.shape}, num_classes: {C}")
    W, b = W.to(device), b.to(device)

    logger.info("Loading data splits...")
    tr = get_loader(run, "train", batch_size=4096)
    va = get_loader(run, "val",   batch_size=4096)
    te = get_loader(run, "test",  batch_size=4096)

    logger.info("Materializing tensors...")
    if hasattr(tr.dataset, "tensors"):
        Xtr, ytr = tr.dataset.tensors[0], tr.dataset.tensors[1]
        Xva, yva = va.dataset.tensors[0], va.dataset.tensors[1]
        Xte, yte = te.dataset.tensors[0], te.dataset.tensors[1]
    else:
        logger.info("Concatenating batches...")
        Xtr = torch.cat([X for X,_ in tr], 0)
        ytr = torch.cat([y for _,y in tr], 0)
        Xva = torch.cat([X for X,_ in va], 0)
        yva = torch.cat([y for _,y in va], 0)
        Xte = torch.cat([X for X,_ in te], 0)
        yte = torch.cat([y for _,y in te], 0)
    
    logger.info(f"Train: {Xtr.shape[0]} samples, Val: {Xva.shape[0]} samples, Test: {Xte.shape[0]} samples")
    logger.info("Moving tensors to device...")
    Xtr, ytr = Xtr.to(device), ytr.to(device)
    Xva, yva = Xva.to(device), yva.to(device)
    Xte, yte = Xte.to(device), yte.to(device)

    logger.info("Computing baseline accuracy and recording original predictions...")
    base = accuracy(Xte, yte, W, b)
    logger.info(f"Baseline test acc (NEC={args.nec}): {base:.4f}")
    
    # Record original predictions for net correction analysis
    logger.info("Recording baseline predictions on test set...")
    original_preds = get_predictions(Xte, W, b)
    logger.info(f"Recorded predictions for {len(original_preds)} test samples")

    logger.info("Starting Type-3 budget curve (concept overrides)...")
    selector = lambda X, topk: rank_uncertain_concepts(X, topk=topk, T=2.0)
    curve = budget_curve_type3(Xte, yte, W, b, selector_fn=selector, topks=tuple(range(1, args.budget+1)), tau=args.tau_concept)
    logger.info(f"Type-3 (concept overrides) acc vs budget: {curve}")

    logger.info("Starting Type-4 weight nudges...")
    W2, b2, log = weight_nudge_eval(Xtr, ytr, Xva, yva, W, b, chosen_indices_fn=selector, tau=args.tau_weight, sample_limit=1000)
    logger.info(f"Type-4: Processed {len(log)} accepted edits out of 1000 attempts")
    base2 = accuracy(Xte, yte, W2, b2)
    logger.info(f"Type-4 nudged test acc: {base2:.4f}  (delta {base2-base:+.4f})  accepted_edits={len(log)}")

    # Compute net corrections
    logger.info("Computing net corrections (corrected - broken)...")
    new_preds = get_predictions(Xte, W2, b2)
    net_corrections = compute_net_corrections(Xte, yte, original_preds, new_preds)
    logger.info(f"Net corrections: {net_corrections['total_corrected']} corrected, "
                f"{net_corrections['total_broken']} broken, "
                f"net: {net_corrections['net_corrections']}")

    logger.info("Saving results...")
    save_json({"baseline_acc": base, "T3_curve": curve, "T4_acc": base2, "T4_log": log,
               "net_corrections": net_corrections,
               "nec": args.nec, "load_path": args.load_path}, outdir, "summary")
    logger.info(f"Results saved to {outdir}/summary.json")

if __name__ == "__main__":
    main()
