# interventions/cli/analyze_confusions.py
import argparse, torch, os
from pathlib import Path
from ..adapters.vlgcbm import VLGCbmRun, load_sparse_head, forward_final, confusion_matrix
from ..selectors.confusion import top_confusions, bucket_indices
from ..selectors.cis import class_pair_impact

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load_path", required=True, help="path with *_concept_features.pt and W_g@NEC=K.pt")
    ap.add_argument("--nec", type=int, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--top_k", type=int, default=10, help="Number of top confusions to analyze")
    ap.add_argument("--concepts", type=str, default=None, help="Path to concepts.txt file")
    ap.add_argument("--classes", type=str, default=None, help="Path to class names file")
    args = ap.parse_args()

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run = VLGCbmRun(load_path=args.load_path, nec=args.nec)
    W, b, C = load_sparse_head(run, device=device)
    
    import os
    from loguru import logger
    fp = run.load_path
    feat_path = os.path.join(fp, f"{args.split}_concept_features.pt")
    label_path = os.path.join(fp, f"{args.split}_concept_labels.pt")
    
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Concept features not found: {feat_path}")
    
    print(f"Loading {args.split} split to {device}...")
    X = torch.load(feat_path, map_location="cpu")
    y = torch.load(label_path, map_location="cpu")
    print(f"  Loaded X shape: {X.shape}, y shape: {y.shape} (on CPU)")
    
    if device != "cpu":
        print(f"  Moving to {device}...")
        X = X.to(device)
        y = y.to(device)

    logits = forward_final(X, W, b)
    p = logits.argmax(1)
    
    cm = confusion_matrix(y, p, C)
    
    # Filter to only actual confusions (true != pred)
    misclassified_mask = (y != p)
    y_mis = y[misclassified_mask]
    p_mis = p[misclassified_mask]
    
    if len(y_mis) == 0:
        print("No misclassifications found!")
        return
    
    pairs = top_confusions(y_mis, p_mis, k=args.top_k)
    
    concepts = None
    if args.concepts:
        concepts_path = Path(args.concepts)
    else:
        concepts_path = Path(args.load_path) / "concepts.txt"
    
    if concepts_path.exists():
        with open(concepts_path, "r") as f:
            concepts = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded {len(concepts)} concept names")
    
    class_names = None
    if args.classes:
        with open(args.classes, "r") as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded {len(class_names)} class names")
    
    print("\n" + "="*80)
    print(f"CONFUSION ANALYSIS: {args.split.upper()} SET")
    print("="*80)
    print(f"Total samples: {len(y)}")
    print(f"Correct: {(y == p).sum().item()} ({100*(y==p).float().mean():.2f}%)")
    print(f"Misclassified: {(y != p).sum().item()} ({100*(y!=p).float().mean():.2f}%)")
    print(f"\nTop {args.top_k} Confusion Pairs (true -> pred, count):")
    print("-"*80)
    
    for rank, ((t, q), count) in enumerate(pairs, 1):
        t_name = class_names[t] if class_names and t < len(class_names) else f"Class_{t}"
        q_name = class_names[q] if class_names and q < len(class_names) else f"Class_{q}"
        print(f"\n{rank}. {t_name} ({t}) -> {q_name} ({q}): {count} samples")
        
        indices = bucket_indices(y, p, (t, q))
        if len(indices) == 0:
            continue
        
        X_pair = X[indices]
        
        g = class_pair_impact(W, t, q)
        top_concepts = torch.topk(g, k=min(10, len(g)), largest=True)
        
        print(f"   Top concepts by CIS score (|W[{t}] - W[{q}]|):")
        for idx, (concept_idx, score) in enumerate(zip(top_concepts.indices, top_concepts.values), 1):
            concept_name = concepts[concept_idx] if concepts and concept_idx < len(concepts) else f"Concept_{concept_idx}"
            avg_activation = X_pair[:, concept_idx].mean().item()
            print(f"      {idx:2d}. {concept_name:30s} (idx {concept_idx:3d}): score={score:.4f}, avg_activation={avg_activation:.4f}")
        
        correct_indices = torch.nonzero((y == p) & (y == t), as_tuple=False).view(-1)
        if len(correct_indices) > 0:
            X_correct = X[correct_indices]
            print(f"   Comparison with correctly classified {t_name} samples ({len(correct_indices)} samples):")
            for idx, concept_idx in enumerate(top_concepts.indices[:5], 1):
                concept_name = concepts[concept_idx] if concepts and concept_idx < len(concepts) else f"Concept_{concept_idx}"
                avg_confused = X_pair[:, concept_idx].mean().item()
                avg_correct = X_correct[:, concept_idx].mean().item()
                diff = avg_correct - avg_confused
                print(f"      {idx}. {concept_name:30s}: confused={avg_confused:.4f}, correct={avg_correct:.4f}, diff={diff:+.4f}")

if __name__ == "__main__":
    main()
