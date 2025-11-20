import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="Path to summary.json")
    ap.add_argument("--out", default=None, help="Output path for plot (default: same dir as summary)")
    args = ap.parse_args()

    with open(args.summary, "r") as f:
        data = json.load(f)

    base = data["baseline_acc"]
    t3_curve = data["T3_curve"]
    t4_acc = data["T4_acc"]
    nec = data.get("nec", "N/A")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ks = sorted([k for k in t3_curve.keys() if isinstance(k, (int, float))])
    accs = [t3_curve[k] for k in ks]

    axes[0].plot(ks, accs, marker="o", linewidth=2, markersize=8, label="Type-3 (Concept Overrides)")
    axes[0].axhline(base, color="gray", linestyle="--", alpha=0.7, label="Baseline")
    axes[0].set_xlabel("Number of Concept Edits (k)", fontsize=11)
    axes[0].set_ylabel("Test Accuracy", fontsize=11)
    axes[0].set_title(f"Type-3 Budget Curve (NEC={nec})", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_ylim([min(accs + [base]) - 0.01, max(accs + [base]) + 0.01])

    for k, acc in zip(ks, accs):
        delta = acc - base
        axes[0].annotate(f"+{delta:.3f}", (k, acc), textcoords="offset points", 
                        xytext=(0, 10), ha="center", fontsize=9)

    axes[1].bar(["Baseline", "Type-4 (Weight Nudges)"], [base, t4_acc], 
                color=["gray", "steelblue"], alpha=0.7, width=0.5)
    axes[1].set_ylabel("Test Accuracy", fontsize=11)
    axes[1].set_title(f"Type-4 Weight Nudges (NEC={nec})", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].set_ylim([min(base, t4_acc) - 0.01, max(base, t4_acc) + 0.01])

    delta_t4 = t4_acc - base
    axes[1].annotate(f"{t4_acc:.4f}\n({delta_t4:+.4f})", 
                    ("Type-4 (Weight Nudges)", t4_acc), 
                    textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10)
    axes[1].annotate(f"{base:.4f}", 
                    ("Baseline", base), 
                    textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10)

    t4_log = data.get("T4_log", [])
    if t4_log:
        axes[1].text(0.5, -0.15, f"Accepted edits: {len(t4_log)}", 
                    transform=axes[1].transAxes, ha="center", fontsize=9, style="italic")

    plt.tight_layout()

    if args.out:
        out_path = args.out
    else:
        out_path = str(Path(args.summary).parent / "intervention_results.png")

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {out_path}")

    print("\n" + "="*60)
    print("INTERVENTION RESULTS SUMMARY")
    print("="*60)
    print(f"Baseline Accuracy: {base:.4f}")
    print(f"\nType-3 (Concept Overrides):")
    for k in ks:
        delta = t3_curve[k] - base
        print(f"  k={k}: {t3_curve[k]:.4f} (delta: {delta:+.4f})")
    print(f"\nType-4 (Weight Nudges):")
    print(f"  Accuracy: {t4_acc:.4f} (delta: {delta_t4:+.4f})")
    print(f"  Accepted edits: {len(t4_log)}")
    print("="*60)

if __name__ == "__main__":
    main()

