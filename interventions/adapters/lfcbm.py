# interventions/adapters/lfcbm.py
# Thin placeholder to mirror the VLG adapter API; fill paths to your LF-CBM tensors.
from dataclasses import dataclass
import os, torch
from torch.utils.data import TensorDataset, DataLoader

@dataclass
class LFCbmRun:
    load_path: str     # point this at the directory with *_concept_features.pt and W/b files
    nec: int | None = None

def _load_split_tensors(run: LFCbmRun, split: str):
    X = torch.load(os.path.join(run.load_path, f"{split}_concept_features.pt"), map_location="cpu")
    y = torch.load(os.path.join(run.load_path, f"{split}_concept_labels.pt"),  map_location="cpu")
    return X, y

def get_loader(run: LFCbmRun, split: str, batch_size=256, num_workers=2, shuffle=False):
    X, y = _load_split_tensors(run, split)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

def load_sparse_head(run: LFCbmRun):
    if run.nec is None:
        raise ValueError("Specify nec to load a precomputed sparse head.")
    W = torch.load(os.path.join(run.load_path, f"W_g@NEC={run.nec}.pt"), map_location="cpu")
    b = torch.load(os.path.join(run.load_path, f"b_g@NEC={run.nec}.pt"), map_location="cpu")
    C, D = W.shape
    return W, b, C

def forward_final(X, W, b): return X @ W.T + b
def predict(X, W, b): return forward_final(X, W, b).argmax(dim=1)
