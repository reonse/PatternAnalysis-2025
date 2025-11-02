import numpy as np
import torch
import torch.nn.functional as F

def one_hot(y: torch.Tensor, C: int) -> torch.Tensor:
    B, H, W = y.shape
    oh = torch.zeros(B, C, H, W, device=y.device, dtype=torch.float32)
    return oh.scatter_(1, y.unsqueeze(1), 1.0)

@torch.no_grad()
def dice_per_class(logits: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> np.ndarray:
    p = F.softmax(logits, dim=1)
    yoh = one_hot(y, p.shape[1])
    num = 2 * (p * yoh).sum(dim=(0, 2, 3))
    den = (p + yoh).sum(dim=(0, 2, 3)) + eps
    return (num / den).detach().cpu().numpy()

class DiceCELoss(torch.nn.Module):
    def __init__(self, n_classes: int = 4, dice_w: float = 0.7, ce_w: float = 0.3, eps: float = 1e-6):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss()
        self.dw, self.cw, self.eps = dice_w, ce_w, eps
        self.C = n_classes
    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, y)
        p = F.softmax(logits, dim=1)
        yoh = one_hot(y, self.C)
        num = 2 * (p * yoh).sum(dim=(0, 2, 3))
        den = (p + yoh).sum(dim=(0, 2, 3)) + self.eps
        dice = 1.0 - (num / den).mean()
        return self.dw * dice + self.cw * ce
