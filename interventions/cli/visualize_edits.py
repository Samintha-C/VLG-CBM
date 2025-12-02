# interventions/cli/visualize_edits.py
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

def load_concepts(concepts_path):
    if concepts_path and Path(concepts_path).exists():
        with open(concepts_path, "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    return None

def visualize_type3_edits(summary_data, concepts=None, output_dir=None):
    """Visualize Type-3 concept override edits"""
    if "T3_edits" not in summary_data:
        print("No Type-3 edit data found")
        return
    
    t3_edits = summary_data["T3_edits"]
    t3_curve = summary_data.get("T3_curve", {})
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Type-3 Concept Override Edits Analysis", fontsize=16, fontweight="bold")
    
    # 1. Concept edit frequency (top concepts edited)
    concept_counts = Counter()
    concept_deltas = defaultdict(list)
    
    for k, edits_list in t3_edits.items():
        for sample_edit in edits_list:
            for ce in sample_edit.get("concept_edits", []):
                cidx = ce["concept_idx"]
                concept_counts[cidx] += 1
                concept_deltas[cidx].append(abs(ce["delta"]))
    
    top_concepts = concept_counts.most_common(20)
    concept_names = [concepts[c] if concepts and c < len(concepts) else f"Concept_{c}" 
                     for c, _ in top_concepts]
    counts = [count for _, count in top_concepts]
    avg_deltas = [np.mean(concept_deltas[c]) for c, _ in top_concepts]
    
    axes[0, 0].barh(range(len(concept_names)), counts)
    axes[0, 0].set_yticks(range(len(concept_names)))
    axes[0, 0].set_yticklabels(concept_names, fontsize=8)
    axes[0, 0].set_xlabel("Number of Edits")
    axes[0, 0].set_title("Most Frequently Edited Concepts")
    axes[0, 0].invert_yaxis()
    
    # 2. Average edit magnitude per concept
    axes[0, 1].barh(range(len(concept_names)), avg_deltas, color='orange')
    axes[0, 1].set_yticks(range(len(concept_names)))
    axes[0, 1].set_yticklabels(concept_names, fontsize=8)
    axes[0, 1].set_xlabel("Average |Delta|")
    axes[0, 1].set_title("Average Edit Magnitude per Concept")
    axes[0, 1].invert_yaxis()
    
    # 3. Accuracy vs budget (k)
    if t3_curve:
        ks = sorted([k for k in t3_curve.keys() if isinstance(k, int)])
        accs = [t3_curve[k] for k in ks]
        axes[1, 0].plot(ks, accs, marker='o', linewidth=2, markersize=8)
        axes[1, 0].axhline(y=t3_curve.get(0, accs[0]), color='r', linestyle='--', label='Baseline')
        axes[1, 0].set_xlabel("Number of Concepts Edited (k)")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].set_title("Type-3 Budget Curve")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
    
    # 4. Edit distribution histogram
    all_deltas = []
    for k, edits_list in t3_edits.items():
        for sample_edit in edits_list:
            for ce in sample_edit.get("concept_edits", []):
                all_deltas.append(ce["delta"])
    
    if all_deltas:
        axes[1, 1].hist(all_deltas, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel("Edit Delta (concept activation change)")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_title("Distribution of Concept Edit Magnitudes")
        axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_dir:
        output_path = Path(output_dir) / "type3_edits_analysis.png"
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Type-3 visualization saved to: {output_path}")
    else:
        plt.show()

def visualize_type4_edits(summary_data, concepts=None, output_dir=None):
    """Visualize Type-4 weight nudge edits"""
    if "T4_log" not in summary_data:
        print("No Type-4 edit data found")
        return
    
    t4_log = summary_data["T4_log"]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Type-4 Weight Nudge Edits Analysis", fontsize=16, fontweight="bold")
    
    # 1. Most nudged concepts (by frequency)
    concept_counts = Counter()
    concept_weight_changes = defaultdict(lambda: {"true": [], "pred": []})
    
    for entry in t4_log:
        for wc in entry.get("weight_changes", []):
            cidx = wc["concept_idx"]
            concept_counts[cidx] += 1
            concept_weight_changes[cidx]["true"].append(wc.get("delta_W_true", 0))
            concept_weight_changes[cidx]["pred"].append(wc.get("delta_W_pred", 0))
    
    top_concepts = concept_counts.most_common(20)
    concept_names = [concepts[c] if concepts and c < len(concepts) else f"Concept_{c}" 
                     for c, _ in top_concepts]
    counts = [count for _, count in top_concepts]
    
    axes[0, 0].barh(range(len(concept_names)), counts)
    axes[0, 0].set_yticks(range(len(concept_names)))
    axes[0, 0].set_yticklabels(concept_names, fontsize=8)
    axes[0, 0].set_xlabel("Number of Weight Nudges")
    axes[0, 0].set_title("Most Frequently Nudged Concepts")
    axes[0, 0].invert_yaxis()
    
    # 2. Average weight change per concept (true vs pred)
    avg_delta_true = [np.mean(concept_weight_changes[c]["true"]) if concept_weight_changes[c]["true"] else 0 
                      for c, _ in top_concepts]
    avg_delta_pred = [np.mean(concept_weight_changes[c]["pred"]) if concept_weight_changes[c]["pred"] else 0 
                      for c, _ in top_concepts]
    
    x = np.arange(len(concept_names))
    width = 0.35
    axes[0, 1].barh(x - width/2, avg_delta_true, width, label='True Class', color='green', alpha=0.7)
    axes[0, 1].barh(x + width/2, avg_delta_pred, width, label='Pred Class', color='red', alpha=0.7)
    axes[0, 1].set_yticks(x)
    axes[0, 1].set_yticklabels(concept_names, fontsize=8)
    axes[0, 1].set_xlabel("Average Weight Change")
    axes[0, 1].set_title("Average Weight Changes (True vs Pred)")
    axes[0, 1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    axes[0, 1].legend()
    axes[0, 1].invert_yaxis()
    
    # 3. Validation accuracy over time (as edits are accepted)
    val_accs = [entry.get("val_acc_after", 0) for entry in t4_log if "val_acc_after" in entry]
    if val_accs:
        axes[1, 0].plot(range(len(val_accs)), val_accs, marker='o', linewidth=2, markersize=4)
        baseline = summary_data.get("baseline_val_acc", val_accs[0] if val_accs else 0)
        axes[1, 0].axhline(y=baseline, color='r', linestyle='--', label='Baseline')
        axes[1, 0].set_xlabel("Accepted Edit Number")
        axes[1, 0].set_ylabel("Validation Accuracy")
        axes[1, 0].set_title("Validation Accuracy During Weight Nudging")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
    
    # 4. Weight change distribution
    all_deltas_true = []
    all_deltas_pred = []
    for entry in t4_log:
        for wc in entry.get("weight_changes", []):
            if "delta_W_true" in wc:
                all_deltas_true.append(wc["delta_W_true"])
            if "delta_W_pred" in wc:
                all_deltas_pred.append(wc["delta_W_pred"])
    
    if all_deltas_true or all_deltas_pred:
        axes[1, 1].hist(all_deltas_true, bins=50, alpha=0.5, label='True Class', color='green', edgecolor='black')
        axes[1, 1].hist(all_deltas_pred, bins=50, alpha=0.5, label='Pred Class', color='red', edgecolor='black')
        axes[1, 1].set_xlabel("Weight Change Delta")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_title("Distribution of Weight Changes")
        axes[1, 1].axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_dir:
        output_path = Path(output_dir) / "type4_edits_analysis.png"
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Type-4 visualization saved to: {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize concept edit details from intervention results")
    parser.add_argument("--summary", type=str, required=True, help="Path to summary.json")
    parser.add_argument("--concepts", type=str, default=None, help="Path to concepts.txt file")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save visualizations")
    args = parser.parse_args()
    
    summary_path = Path(args.summary)
    with open(summary_path, "r") as f:
        summary_data = json.load(f)
    
    output_dir = args.output_dir or summary_path.parent
    
    concepts = load_concepts(args.concepts)
    if not concepts:
        # Try to find concepts.txt in the same directory
        concepts_path = summary_path.parent / "concepts.txt"
        if not concepts_path.exists() and "load_path" in summary_data:
            concepts_path = Path(summary_data["load_path"]) / "concepts.txt"
        concepts = load_concepts(concepts_path)
    
    if concepts:
        print(f"Loaded {len(concepts)} concept names")
    else:
        print("Warning: No concept names found. Using indices.")
    
    print("\nGenerating visualizations...")
    visualize_type3_edits(summary_data, concepts, output_dir)
    visualize_type4_edits(summary_data, concepts, output_dir)
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()

