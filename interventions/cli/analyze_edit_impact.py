# interventions/cli/analyze_edit_impact.py
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

def load_concepts(concepts_path):
    if concepts_path and Path(concepts_path).exists():
        with open(concepts_path, "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    return None

def analyze_concept_edit_impact(summary_data, X_test, y_test, W_original, W_modified, concepts=None):
    """
    Analyze which concept edits help vs hurt by correlating edits with outcomes.
    
    Returns metrics per concept:
    - Success rate (corrections / total edits)
    - Net impact (corrections - breakages)
    - Average edit magnitude for successful vs failed edits
    - Relationship between edit characteristics and outcomes
    """
    from ..adapters.vlgcbm import get_predictions
    
    # Get predictions before and after
    pred_before = get_predictions(X_test, W_original, torch.zeros(W_original.shape[0], device=W_original.device))
    pred_after = get_predictions(X_test, W_modified, torch.zeros(W_modified.shape[0], device=W_modified.device))
    
    # Find changed predictions
    changed_mask = pred_before != pred_after
    changed_indices = torch.nonzero(changed_mask, as_tuple=False).view(-1)
    
    # Track per-concept impact
    concept_metrics = defaultdict(lambda: {
        "total_edits": 0,
        "corrections": 0,
        "breakages": 0,
        "edit_magnitudes": [],
        "successful_edit_magnitudes": [],
        "failed_edit_magnitudes": [],
        "weight_changes_true": [],
        "weight_changes_pred": [],
        "activations": []
    })
    
    # Analyze Type-4 weight nudges
    t4_log = summary_data.get("T4_log", [])
    for entry in t4_log:
        for wc in entry.get("weight_changes", []):
            cidx = wc.get("concept_idx", -1)
            if cidx < 0:
                continue
            
            concept_metrics[cidx]["total_edits"] += 1
            concept_metrics[cidx]["weight_changes_true"].append(wc.get("delta_W_true", 0))
            concept_metrics[cidx]["weight_changes_pred"].append(wc.get("delta_W_pred", 0))
            concept_metrics[cidx]["activations"].append(wc.get("concept_activation", 0))
    
    # Correlate edits with outcomes
    # For each changed sample, find which concepts were edited
    for idx in changed_indices:
        gt = int(y_test[idx])
        orig_pred = int(pred_before[idx])
        new_pred = int(pred_after[idx])
        
        # Find which concepts were nudged for this class pair
        for entry in t4_log:
            if entry.get("true_class") == gt and entry.get("pred_class") == orig_pred:
                for wc in entry.get("weight_changes", []):
                    cidx = wc.get("concept_idx", -1)
                    if cidx < 0:
                        continue
                    
                    if gt == new_pred:
                        concept_metrics[cidx]["corrections"] += 1
                        concept_metrics[cidx]["successful_edit_magnitudes"].append(
                            abs(wc.get("delta_W_true", 0)) + abs(wc.get("delta_W_pred", 0))
                        )
                    elif orig_pred == new_pred and gt != orig_pred:
                        concept_metrics[cidx]["breakages"] += 1
                        concept_metrics[cidx]["failed_edit_magnitudes"].append(
                            abs(wc.get("delta_W_true", 0)) + abs(wc.get("delta_W_pred", 0))
                        )
    
    # Compute aggregate metrics
    results = {}
    for cidx, metrics in concept_metrics.items():
        if metrics["total_edits"] == 0:
            continue
        
        results[cidx] = {
            "total_edits": metrics["total_edits"],
            "corrections": metrics["corrections"],
            "breakages": metrics["breakages"],
            "net_impact": metrics["corrections"] - metrics["breakages"],
            "success_rate": metrics["corrections"] / metrics["total_edits"] if metrics["total_edits"] > 0 else 0,
            "avg_edit_magnitude": np.mean(metrics["weight_changes_true"] + metrics["weight_changes_pred"]) if metrics["weight_changes_true"] or metrics["weight_changes_pred"] else 0,
            "avg_successful_magnitude": np.mean(metrics["successful_edit_magnitudes"]) if metrics["successful_edit_magnitudes"] else 0,
            "avg_failed_magnitude": np.mean(metrics["failed_edit_magnitudes"]) if metrics["failed_edit_magnitudes"] else 0,
            "avg_activation": np.mean(metrics["activations"]) if metrics["activations"] else 0,
        }
    
    return results

def analyze_type3_edit_impact(summary_data, X_test, y_test, W, b, concepts=None):
    """
    Analyze Type-3 concept override edits and their impact.
    """
    from ..adapters.vlgcbm import get_predictions, forward_final
    
    t3_edits = summary_data.get("T3_edits", {})
    if not t3_edits:
        return {}
    
    concept_metrics = defaultdict(lambda: {
        "total_edits": 0,
        "edit_magnitudes": [],
        "old_activations": [],
        "new_activations": [],
        "weight_diffs": [],
        "edit_directions": []  # positive vs negative deltas
    })
    
    # Analyze edits by concept
    for k, edits_list in t3_edits.items():
        for sample_edit in edits_list:
            for ce in sample_edit.get("concept_edits", []):
                cidx = ce.get("concept_idx", -1)
                if cidx < 0:
                    continue
                
                concept_metrics[cidx]["total_edits"] += 1
                concept_metrics[cidx]["edit_magnitudes"].append(abs(ce.get("delta", 0)))
                concept_metrics[cidx]["old_activations"].append(ce.get("old_activation", 0))
                concept_metrics[cidx]["new_activations"].append(ce.get("new_activation", 0))
                concept_metrics[cidx]["weight_diffs"].append(abs(ce.get("weight_diff", 0)))
                concept_metrics[cidx]["edit_directions"].append(1 if ce.get("delta", 0) > 0 else -1)
    
    # Compute aggregate metrics
    results = {}
    for cidx, metrics in concept_metrics.items():
        if metrics["total_edits"] == 0:
            continue
        
        results[cidx] = {
            "total_edits": metrics["total_edits"],
            "avg_edit_magnitude": np.mean(metrics["edit_magnitudes"]) if metrics["edit_magnitudes"] else 0,
            "max_edit_magnitude": np.max(metrics["edit_magnitudes"]) if metrics["edit_magnitudes"] else 0,
            "avg_old_activation": np.mean(metrics["old_activations"]) if metrics["old_activations"] else 0,
            "avg_new_activation": np.mean(metrics["new_activations"]) if metrics["new_activations"] else 0,
            "avg_weight_diff": np.mean(metrics["weight_diffs"]) if metrics["weight_diffs"] else 0,
            "positive_edits_ratio": sum(1 for d in metrics["edit_directions"] if d > 0) / len(metrics["edit_directions"]) if metrics["edit_directions"] else 0,
        }
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Analyze which concept edits help vs hurt")
    parser.add_argument("--summary", type=str, required=True, help="Path to summary.json")
    parser.add_argument("--concepts", type=str, default=None, help="Path to concepts.txt")
    parser.add_argument("--load_path", type=str, default=None, help="Model load path (for loading test data)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for metrics")
    args = parser.parse_args()
    
    with open(args.summary, "r") as f:
        summary_data = json.load(f)
    
    concepts = load_concepts(args.concepts)
    
    print("="*80)
    print("CONCEPT EDIT IMPACT ANALYSIS")
    print("="*80)
    
    # Analyze Type-3 edits
    print("\nType-3 (Concept Overrides) Edit Analysis:")
    print("-"*80)
    t3_metrics = analyze_type3_edit_impact(summary_data, None, None, None, None, concepts)
    
    if t3_metrics:
        # Sort by total edits
        sorted_t3 = sorted(t3_metrics.items(), key=lambda x: x[1]["total_edits"], reverse=True)[:20]
        print(f"\nTop 20 Most Edited Concepts:")
        for cidx, metrics in sorted_t3:
            name = concepts[cidx] if concepts and cidx < len(concepts) else f"Concept_{cidx}"
            print(f"  [{cidx}] {name}:")
            print(f"    Total edits: {metrics['total_edits']}")
            print(f"    Avg magnitude: {metrics['avg_edit_magnitude']:.4f}")
            print(f"    Avg old activation: {metrics['avg_old_activation']:.4f}")
            print(f"    Avg new activation: {metrics['avg_new_activation']:.4f}")
            print(f"    Positive edits ratio: {metrics['positive_edits_ratio']:.2%}")
    
    # Analyze Type-4 edits (requires loading model and data)
    print("\n" + "="*80)
    print("Type-4 (Weight Nudges) Impact Analysis:")
    print("-"*80)
    print("Note: Full Type-4 impact analysis requires loading test data and model weights.")
    print("This would show which weight nudges led to corrections vs breakages.")
    
    # For now, analyze from T4_log
    t4_log = summary_data.get("T4_log", [])
    if t4_log:
        concept_stats = defaultdict(lambda: {"edits": 0, "weight_changes": []})
        for entry in t4_log:
            for wc in entry.get("weight_changes", []):
                cidx = wc.get("concept_idx", -1)
                if cidx >= 0:
                    concept_stats[cidx]["edits"] += 1
                    concept_stats[cidx]["weight_changes"].append({
                        "delta_true": wc.get("delta_W_true", 0),
                        "delta_pred": wc.get("delta_W_pred", 0),
                        "activation": wc.get("concept_activation", 0)
                    })
        
        print(f"\nTop 20 Most Nudged Concepts (by frequency):")
        sorted_t4 = sorted(concept_stats.items(), key=lambda x: x[1]["edits"], reverse=True)[:20]
        for cidx, stats in sorted_t4:
            name = concepts[cidx] if concepts and cidx < len(concepts) else f"Concept_{cidx}"
            avg_delta_t = np.mean([abs(wc["delta_true"]) for wc in stats["weight_changes"]])
            avg_delta_p = np.mean([abs(wc["delta_pred"]) for wc in stats["weight_changes"]])
            avg_act = np.mean([wc["activation"] for wc in stats["weight_changes"]])
            print(f"  [{cidx}] {name}:")
            print(f"    Total nudges: {stats['edits']}")
            print(f"    Avg |ΔW_true|: {avg_delta_t:.6f}")
            print(f"    Avg |ΔW_pred|: {avg_delta_p:.6f}")
            print(f"    Avg activation: {avg_act:.4f}")
    
    # Save metrics if output specified
    if args.output:
        output_data = {
            "type3_metrics": {str(k): v for k, v in t3_metrics.items()},
            "type4_stats": {str(k): {"edits": v["edits"], "avg_delta_true": np.mean([abs(wc["delta_true"]) for wc in v["weight_changes"]]),
                                    "avg_delta_pred": np.mean([abs(wc["delta_pred"]) for wc in v["weight_changes"]]),
                                    "avg_activation": np.mean([wc["activation"] for wc in v["weight_changes"]])}
                           for k, v in concept_stats.items()}
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nMetrics saved to: {args.output}")

if __name__ == "__main__":
    main()

