# interventions/adapters/vlgcbm.py
from dataclasses import dataclass
import os, glob, torch
from torch.utils.data import TensorDataset, DataLoader
from loguru import logger

@dataclass
class VLGCbmRun:
    load_path: str     # e.g., /sc-cbint-vol/saved_models/vlg/cifar100
    nec: int | None = None  # if None, you can refit; else load W_g@NEC=nec

def _load_split_tensors(run: VLGCbmRun, split: str, device="cpu"):
    fp = run.load_path
    feat_path = os.path.join(fp, f"{split}_concept_features.pt")
    label_path = os.path.join(fp, f"{split}_concept_labels.pt")
    
    if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Concept features not found: {feat_path}")
    
    logger.info(f"Loading {split} split: {os.path.basename(feat_path)} to {device}")
    # Load directly to the specified device to avoid double memory usage
    X = torch.load(feat_path, map_location=device)
    y = torch.load(label_path, map_location=device)
    logger.info(f"  Loaded X shape: {X.shape}, y shape: {y.shape}")
    return X, y

def get_loader(run: VLGCbmRun, split: str, batch_size=256, num_workers=2, shuffle=False, device="cpu"):
    X, y = _load_split_tensors(run, split, device=device)
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=False)

def split_validation_set(run: VLGCbmRun, intervention_ratio=0.5, seed=42):
    """
    Split validation set into two parts:
    - val_intervention: For running interventions (finding mistakes, applying fixes)
    - val_eval: For evaluating intervened model (generalization check)
    
    Args:
        run: VLGCbmRun instance
        intervention_ratio: Fraction of val set to use for interventions (default 0.5)
        seed: Random seed for reproducibility
    
    Returns:
        (X_val_int, y_val_int, X_val_eval, y_val_eval): Split validation sets
    """
    import torch
    torch.manual_seed(seed)
    
    X_val, y_val = _load_split_tensors(run, "val")
    n = len(X_val)
    n_int = int(n * intervention_ratio)
    
    # Random permutation for splitting
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    int_indices = indices[:n_int]
    eval_indices = indices[n_int:]
    
    X_val_int = X_val[int_indices]
    y_val_int = y_val[int_indices]
    X_val_eval = X_val[eval_indices]
    y_val_eval = y_val[eval_indices]
    
    logger.info(f"Split validation set: {n} samples")
    logger.info(f"  Intervention set: {len(X_val_int)} samples ({intervention_ratio:.1%})")
    logger.info(f"  Evaluation set: {len(X_val_eval)} samples ({1-intervention_ratio:.1%})")
    
    return X_val_int, y_val_int, X_val_eval, y_val_eval

def list_available_nec(load_path: str):
    ws = sorted(glob.glob(os.path.join(load_path, "W_g@NEC=*.pt")))
    def _nec(p): return int(os.path.basename(p).split("=")[1].split(".")[0])
    return sorted(set(_nec(p) for p in ws))

def load_sparse_head(run: VLGCbmRun, device="cuda"):
    """Returns W [C x D], b [C], and inferred num_classes."""
    if run.nec is None:
        raise ValueError("Specify nec to load a precomputed sparse head.")
    fp = run.load_path
    w_path = os.path.join(fp, f"W_g@NEC={run.nec}.pt")
    b_path = os.path.join(fp, f"b_g@NEC={run.nec}.pt")
    logger.info(f"Loading sparse head: W from {os.path.basename(w_path)}, b from {os.path.basename(b_path)} to {device}")
    W = torch.load(w_path, map_location=device)
    b = torch.load(b_path, map_location=device)
    C, D = W.shape
    logger.info(f"  Loaded W shape: {W.shape}, b shape: {b.shape}, num_classes: {C}")
    return W, b, C

def forward_final(X, W, B):
    return X @ W.T + B  # [N, D] x [D, C]^T = [N, C]

def predict(X, W, b):
    return forward_final(X, W, b).argmax(dim=1)

def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int):
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm
