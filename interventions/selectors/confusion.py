# interventions/selectors/confusion.py
from collections import Counter, defaultdict
import torch

def top_confusions(y_true: torch.Tensor, y_pred: torch.Tensor, k: int = 5):
    pairs = list(zip(y_true.tolist(), y_pred.tolist()))
    counts = Counter(pairs)
    return counts.most_common(k)

def bucket_indices(y_true: torch.Tensor, y_pred: torch.Tensor, pair):
    t, p = pair
    mask = (y_true == t) & (y_pred == p)
    return torch.nonzero(mask, as_tuple=False).view(-1)
