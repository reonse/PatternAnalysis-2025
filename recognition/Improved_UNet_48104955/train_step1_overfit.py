# train_step1_overfit.py
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time
from torch.utils.data import DataLoader
from dataset_simple import OASISDataset
from model_simple import UNetImproved2D

def one_hot(y, C):
    B,H,W = y.shape
    oh = torch.zeros(B, C, H, W, device=y.device, dtype=torch.float32)
    return oh.scatter_(1, y.unsqueeze(1), 1.0)

class DiceCELoss(nn.Module):
    def __init__(self, n_classes=4, dice_w=0.7, ce_w=0.3, eps=1e-6):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dw, self.cw, self.eps = dice_w, ce_w, eps
        self.C = n_classes
    def forward(self, logits, y):
        ce = self.ce(logits, y)
        p = F.softmax(logits, dim=1)
        yoh = one_hot(y, self.C)
        num = 2 * (p*yoh).sum(dim=(0,2,3))
        den = (p + yoh).sum(dim=(0,2,3)) + self.eps
        dice = 1.0 - (num/den).mean()
        return self.dw*dice + self.cw*ce

@torch.no_grad()
def dice_per_class(logits, y, eps=1e-6):
    p = F.softmax(logits, dim=1)
    yoh = one_hot(y, p.shape[1])
    num = 2 * (p*yoh).sum(dim=(0,2,3))
    den = (p + yoh).sum(dim=(0,2,3)) + eps
    return (num/den).detach().cpu().numpy()

if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # small subset for overfit (e.g., 32 slices)
    ds = OASISDataset("data", "train")
    idxs = list(range(min(32, len(ds))))
    subset = torch.utils.data.Subset(ds, idxs)
    dl = DataLoader(subset, batch_size=8, shuffle=True)

    m = UNetImproved2D().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-5)
    crit = DiceCELoss()

    best = -1.0
    for ep in range(1, 16):
        m.train(); losses=[]
        t0 = time.time()
        for x,y in dl:
            x,y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            logits = m(x)
            loss = crit(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        # quick eval on the same subset (we WANT it to increase here)
        m.eval(); acc = np.zeros(4); n=0
        with torch.no_grad():
            for x,y in dl:
                x,y = x.to(dev), y.to(dev)
                acc += dice_per_class(m(x), y); n+=1
        dpc = acc/max(1,n); mean_dice = dpc.mean()
        print(f"[{ep:02d}] loss {np.mean(losses):.4f} | overfit mean Dice {mean_dice:.3f} | {time.time()-t0:.1f}s")
        best = max(best, mean_dice)
    print("Best mean Dice (overfit subset):", best)
