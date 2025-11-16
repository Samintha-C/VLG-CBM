# interventions/cli/run_interventions.py
import argparse, torch
from ..adapters.vlgcbm import VLGCbmRun, get_loader, load_sparse_head
from ..selectors.entropy import rank_uncertain_concepts
from ..selectors.confusion import top_confusions, bucket_indices
from ..evaluate.sweep import budget_curve_type3, weight_nudge_eval, accuracy
from ..evaluate.report import stamp_dir, save_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", choices=["vlg"], default="vlg")
    ap.add_argument("--load_path", required=True)
    ap.add_argument("--nec", type=int, required=True)
    ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--tau_concept", type=float, default=2.0)
    ap.add_argument("--tau_weight", type=float, default=1e-2)
    args = ap.parse_args()

    outdir = stamp_dir()
    run = VLGCbmRun(load_path=args.load_path, nec=args.nec)
    W, b, C = load_sparse_head(run)

    # Load splits
    tr = get_loader(run, "train", batch_size=4096)
    va = get_loader(run, "val",   batch_size=4096)
    te = get_loader(run, "test",  batch_size=4096)

    # Materialize tensors for eval simplicity
    Xtr, ytr = next(iter(tr.dataset.tensors)), tr.dataset.tensors[1] if hasattr(tr.dataset, "tensors") else None
    Xva, yva = next(iter(va.dataset.tensors)), va.dataset.tensors[1] if hasattr(va.dataset, "tensors") else None
    Xte, yte = next(iter(te.dataset.tensors)), te.dataset.tensors[1] if hasattr(te.dataset, "tensors") else None
    # If DataLoader has no .tensors, concatenate (sane for our sizes)
    if ytr is None:
        Xtr = torch.cat([X for X,_ in tr], 0); ytr = torch.cat([y for _,y in tr], 0)
        Xva = torch.cat([X for X,_ in va], 0); yva = torch.cat([y for _,y in va], 0)
        Xte = torch.cat([X for X,_ in te], 0); yte = torch.cat([y for _,y in te], 0)

    # Baseline acc
    base = accuracy(Xte, yte, W, b)
    print(f"Baseline test acc (NEC={args.nec}): {base:.4f}")

    # ---- Type-3 budget curve on TEST (UCP/entropy selection)
    selector = lambda X, topk: rank_uncertain_concepts(X, topk=topk, T=2.0)
    curve = budget_curve_type3(Xte, yte, W, b, selector_fn=selector, topks=tuple(range(1, args.budget+1)), tau=args.tau_concept)
    print("Type-3 (concept overrides) acc vs budget:", curve)

    # ---- Type-4 tiny weight nudges (accept only if val acc non-decreasing)
    W2, b2, log = weight_nudge_eval(Xtr, ytr, Xva, yva, W, b, chosen_indices_fn=selector, tau=args.tau_weight, sample_limit=1000)
    base2 = accuracy(Xte, yte, W2, b2)
    print(f"Type-4 nudged test acc: {base2:.4f}  (delta {base2-base:+.4f})  accepted_edits={len(log)}")

    # Save report
    save_json({"baseline_acc": base, "T3_curve": curve, "T4_acc": base2, "T4_log": log,
               "nec": args.nec, "load_path": args.load_path}, outdir, "summary")

if __name__ == "__main__":
    main()
