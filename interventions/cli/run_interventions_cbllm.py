#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
import torch
from loguru import logger

# Add interventions to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from interventions.adapters.cbllm import CBLLMRun, get_loader, load_sparse_head
from interventions.evaluate.sweep import accuracy, budget_curve_type3, weight_nudge_eval, compute_net_corrections, get_predictions
from interventions.selectors.entropy import rank_uncertain_concepts
from interventions.selectors.cis import rank_by_cis
from interventions.evaluate.report import save_report, create_output_dir

def main():
    parser = argparse.ArgumentParser(description="Run interventions on CB-LLM")
    parser.add_argument("--load_path", type=str, required=True,
                        help="Path to CB-LLM model directory (e.g., 'mpnet_acs/SetFit_sst2/roberta_cbm')")
    parser.add_argument("--sparse", action="store_true", default=True,
                        help="Use sparse weights (default: True)")
    parser.add_argument("--backbone", type=str, default="roberta", choices=["roberta", "gpt2"],
                        help="Backbone model type")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (auto-inferred from load_path if not provided)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (default: cuda)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for data loading")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max sequence length for tokenization")
    parser.add_argument("--selector", type=str, default="entropy", choices=["entropy", "cis"],
                        help="Concept selector (default: entropy)")
    parser.add_argument("--topks", type=int, nargs="+", default=[1, 2, 3],
                        help="Number of concepts to edit for Type-3 budget curve")
    parser.add_argument("--tau", type=float, default=2.0,
                        help="Budget constraint for concept overrides")
    parser.add_argument("--weight_tau", type=float, default=1e-2,
                        help="Budget constraint for weight nudges")
    parser.add_argument("--sample_limit", type=int, default=1000,
                        help="Limit number of samples for weight nudging")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: auto-generated)")
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("CB-LLM INTERVENTION PIPELINE")
    logger.info("="*70)
    
    run = CBLLMRun(
        load_path=args.load_path,
        sparse=args.sparse,
        backbone=args.backbone,
        dataset=args.dataset,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device
    )
    
    output_dir = create_output_dir(args.output_dir, prefix="cbllm_interventions")
    logger.info(f"Output directory: {output_dir}")
    
    logger.info("Loading sparse head weights...")
    W, b, num_classes = load_sparse_head(run, device=args.device)
    
    logger.info("Loading data splits...")
    logger.info("="*70)
    logger.info("DATA SPLIT STRATEGY:")
    logger.info("  Train: Training data (not used for interventions) - keeping on CPU to save GPU memory")
    logger.info("  Val: Find mistakes, apply interventions (concept recalibration) - on GPU")
    logger.info("  Test: Pure evaluation (unseen, not used in any intervention decisions) - on GPU")
    logger.info("="*70)
    
    tr = get_loader(run, "train", batch_size=args.batch_size, device="cpu")
    
    try:
        va = get_loader(run, "val", batch_size=args.batch_size, device=args.device)
        has_val = True
        logger.info("Found validation set")
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"Validation set not available: {e}")
        has_val = False
    
    te = get_loader(run, "test", batch_size=args.batch_size, device=args.device)
    
    logger.info("Materializing tensors...")
    if hasattr(tr.dataset, "tensors"):
        Xtr, ytr = tr.dataset.tensors[0], tr.dataset.tensors[1]
        if has_val:
            Xva, yva = va.dataset.tensors[0], va.dataset.tensors[1]
        Xte, yte = te.dataset.tensors[0], te.dataset.tensors[1]
    else:
        logger.info("Concatenating batches...")
        Xtr = torch.cat([X for X,_ in tr], 0)
        ytr = torch.cat([y for _,y in tr], 0)
        if has_val:
            Xva = torch.cat([X for X,_ in va], 0)
            yva = torch.cat([y for _,y in va], 0)
        Xte = torch.cat([X for X,_ in te], 0)
        yte = torch.cat([y for _,y in te], 0)
    
    if not has_val:
        logger.info("="*70)
        logger.info("SPLITTING TEST SET:")
        logger.info("  Test set will be split into:")
        logger.info("    - Val (intervention): For finding mistakes and applying interventions")
        logger.info("    - Test: Unseen evaluation set (split from test)")
        logger.info("="*70)
        
        n_test = len(Xte)
        n_int = n_test // 2
        n_eval = n_test - n_int
        
        torch.manual_seed(42)
        indices = torch.randperm(n_test, device=args.device)
        int_indices = indices[:n_int]
        eval_indices = indices[n_int:]
        
        Xva = Xte[int_indices]
        yva = yte[int_indices]
        Xte = Xte[eval_indices]
        yte = yte[eval_indices]
        
        logger.info(f"Split test set ({n_test} samples):")
        logger.info(f"  Val (intervention): {len(Xva)} samples")
        logger.info(f"  Test (from test split): {len(Xte)} samples")
    else:
        logger.info("Using separate validation and test sets")
    
    logger.info(f"Train: {Xtr.shape[0]} samples (on CPU, not used for interventions)")
    logger.info(f"Val: {Xva.shape[0]} samples (for interventions, on {args.device})")
    logger.info(f"Test: {Xte.shape[0]} samples (unseen, for evaluation only, on {args.device})")
    
    if Xva.device != torch.device(args.device):
        logger.info("Moving val to device...")
        Xva, yva = Xva.to(args.device), yva.to(args.device)
    if Xte.device != torch.device(args.device):
        logger.info("Moving test to device...")
        Xte, yte = Xte.to(args.device), yte.to(args.device)
    
    logger.info("Computing baseline accuracy and recording original predictions...")
    base_val = accuracy(Xva, yva, W, b)
    base_test = accuracy(Xte, yte, W, b)
    logger.info(f"Baseline val acc: {base_val:.4f}")
    logger.info(f"Baseline test acc: {base_test:.4f}")
    
    original_preds_test = get_predictions(Xte, W, b)
    
    if args.selector == "entropy":
        selector = lambda X, topk: rank_uncertain_concepts(X, topk=topk)
    elif args.selector == "cis":
        selector = lambda X, topk: rank_by_cis(X, W, yva, get_predictions(X, W, b), topk=topk)
    else:
        raise ValueError(f"Unknown selector: {args.selector}")
    
    logger.info("="*70)
    logger.info("TYPE-3 INTERVENTIONS (Concept Overrides)")
    logger.info("  Running on validation set for holistic impact analysis")
    logger.info("  This matches manual_weight_editing.ipynb: see holistic impact of interventions")
    curve = budget_curve_type3(Xva, yva, W, b, selector_fn=selector, 
                               topks=tuple(args.topks), tau=args.tau)
    
    logger.info("="*70)
    logger.info("TYPE-4 INTERVENTIONS (Weight Nudges)")
    logger.info("  Finding mistakes in validation set, applying weight nudges")
    logger.info("  Accepting nudges if validation accuracy doesn't drop")
    W_new, b_new, nudge_stats = weight_nudge_eval(
        X_train=Xva, y_train=yva, X_val=Xva, y_val=yva, W=W, b=b,
        chosen_indices_fn=selector,
        tau=args.weight_tau, sample_limit=args.sample_limit
    )
    
    logger.info("="*70)
    logger.info("EVALUATING ON UNSEEN TEST SET")
    logger.info("  Computing net corrections (corrected - broken instances)")
    new_preds_test = get_predictions(Xte, W_new, b_new)
    net_corrections = compute_net_corrections(Xte, yte, original_preds_test, new_preds_test)
    
    logger.info("="*70)
    logger.info("FINAL RESULTS:")
    logger.info(f"  Test accuracy before: {net_corrections['accuracy_before']:.4f}")
    logger.info(f"  Test accuracy after: {net_corrections['accuracy_after']:.4f}")
    logger.info(f"  Accuracy delta: {net_corrections['accuracy_delta']:+.4f}")
    logger.info(f"  Total corrected: {net_corrections['total_corrected']}")
    logger.info(f"  Total broken: {net_corrections['total_broken']}")
    logger.info(f"  Net corrections: {net_corrections['net_corrections']}")
    logger.info("="*70)
    
    summary = {
        "load_path": args.load_path,
        "sparse": args.sparse,
        "backbone": args.backbone,
        "selector": args.selector,
        "baseline_val_acc": float(base_val),
        "baseline_test_acc": float(base_test),
        "type3_budget_curve": {int(k): float(v) for k, v in curve.items()},
        "type4_nudge_stats": {k: float(v) if isinstance(v, (int, float)) else v for k, v in nudge_stats.items()},
        "net_corrections": {k: float(v) if isinstance(v, torch.Tensor) else (int(v) if isinstance(v, (int, float)) else v) 
                           for k, v in net_corrections.items()},
    }
    
    save_report(summary, output_dir)
    logger.info(f"Results saved to {output_dir}/summary.json")

if __name__ == "__main__":
    main()

