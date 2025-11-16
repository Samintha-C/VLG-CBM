# interventions/selectors/entropy.py
import torch

def sigmoid_entropy(x: torch.Tensor, T: float = 2.0):
    q = torch.sigmoid(x / T).clamp(1e-6, 1-1e-6)
    return -(q*torch.log(q) + (1-q)*torch.log(1-q))  # [N, D]

def rank_uncertain_concepts(features: torch.Tensor, topk: int = 1, T: float = 2.0):
    """
    UCP baseline (Shin): per-sample rank by concept uncertainty (entropy).
    Returns indices: LongTensor [N, topk]
    """
    H = sigmoid_entropy(features, T=T)
    idx = torch.topk(H, k=topk, dim=1, largest=True).indices
    return idx
