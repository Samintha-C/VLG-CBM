# interventions/replay/memory.py
import torch

class EditMemory:
    """
    Very small cosine-KNN memory on top-K concept dims.
    Stores: (fingerprint: Tensor[D], patch: dict) pairs.
    """
    def __init__(self, topk=16):
        self.topk = topk
        self.bank = []  # list of (fp, patch)

    def _fingerprint(self, x: torch.Tensor):
        idx = torch.topk(x.abs(), k=min(self.topk, x.numel()), largest=True).indices
        fp = torch.zeros_like(x)
        fp[idx] = x[idx]
        return fp / (fp.norm() + 1e-8)

    def add(self, x: torch.Tensor, patch: dict):
        self.bank.append((self._fingerprint(x), patch))

    def query(self, x: torch.Tensor, k=1):
        if not self.bank: return []
        q = self._fingerprint(x)
        sims = torch.tensor([ (q * fp).sum().item() for fp,_ in self.bank ])
        top = torch.topk(sims, k=min(k, len(self.bank))).indices.tolist()
        return [self.bank[i][1] for i in top]
