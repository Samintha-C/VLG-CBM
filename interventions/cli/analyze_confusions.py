# interventions/cli/analyze_confusions.py
import argparse, torch
from ..adapters.vlgcbm import VLGCbmRun, get_loader, load_sparse_head, forward_final, confusion_matrix
from ..selectors.confusion import top_confusions

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load_path", required=True, help="path with *_concept_features.pt and W_g@NEC=K.pt")
    ap.add_argument("--nec", type=int, required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    run = VLGCbmRun(load_path=args.load_path, nec=args.nec)
    W, b, C = load_sparse_head(run)
    loader = get_loader(run, args.split, batch_size=4096)

    ys, ps = [], []
    for X, y in loader:
        logits = forward_final(X, W, b)
        p = logits.argmax(1)
        ys.append(y); ps.append(p)
    y = torch.cat(ys); p = torch.cat(ps)
    cm = confusion_matrix(y, p, C)
    pairs = top_confusions(y, p, k=10)

    print("Top confusions (true->pred, count):")
    for (t, q), c in pairs:
        print(f"{t:3d} -> {q:3d} : {c}")
    print("\nShape of CM:", tuple(cm.shape))

if __name__ == "__main__":
    main()
