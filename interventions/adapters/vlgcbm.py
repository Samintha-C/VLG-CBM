# interventions/adapters/vlgcbm.py
from dataclasses import dataclass
import os, glob, torch
from torch.utils.data import TensorDataset, DataLoader

@dataclass
class VLGCbmRun:
    load_path: str     # e.g., /sc-cbint-vol/saved_models/vlg/cifar100
    nec: int | None = None  # if None, you can refit; else load W_g@NEC=nec

def _load_split_tensors(run: VLGCbmRun, split: str):
    fp = run.load_path
    feat_path = os.path.join(fp, f"{split}_concept_features.pt")
    label_path = os.path.join(fp, f"{split}_concept_labels.pt")
    
    if not os.path.exists(feat_path):
        if split == "test":
            # Fallback to val if test doesn't exist
            feat_path = os.path.join(fp, "val_concept_features.pt")
            label_path = os.path.join(fp, "val_concept_labels.pt")
            if not os.path.exists(feat_path):
                raise FileNotFoundError(f"Neither test nor val concept features found in {fp}")
            import warnings
            warnings.warn(f"test_concept_features.pt not found, using val_concept_features.pt instead")
        else:
            raise FileNotFoundError(f"Concept features not found: {feat_path}")
    
    X = torch.load(feat_path, map_location="cpu")
    y = torch.load(label_path, map_location="cpu")
    return X, y

def get_loader(run: VLGCbmRun, split: str, batch_size=256, num_workers=2, shuffle=False):
    X, y = _load_split_tensors(run, split)
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=False)

def list_available_nec(load_path: str):
    ws = sorted(glob.glob(os.path.join(load_path, "W_g@NEC=*.pt")))
    def _nec(p): return int(os.path.basename(p).split("=")[1].split(".")[0])
    return sorted(set(_nec(p) for p in ws))

def load_sparse_head(run: VLGCbmRun):
    """Returns W [C x D], b [C], and inferred num_classes."""
    if run.nec is None:
        raise ValueError("Specify nec to load a precomputed sparse head.")
    fp = run.load_path
    W = torch.load(os.path.join(fp, f"W_g@NEC={run.nec}.pt"), map_location="cpu")
    b = torch.load(os.path.join(fp, f"b_g@NEC={run.nec}.pt"), map_location="cpu")
    C, D = W.shape
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
