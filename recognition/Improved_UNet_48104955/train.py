from pathlib import Path
import argparse, time, numpy as np, torch
from torch.utils.data import DataLoader
from dataset import OASISDataset
from modules import UNetImproved2D
from utils import DiceCELoss, dice_per_class

def make_loader(root: str, split: str, batch: int, shuffle: bool) -> DataLoader:
    ds = OASISDataset(root, split)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=4, pin_memory=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--out", type=str, default=".")
    ap.add_argument("--test", action="store_true", help="run test evaluation only")
    ap.add_argument("--ckpt", type=str, default="best_unet.pth")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.test:
        model = UNetImproved2D(in_ch=1, n_classes=4, base=args.base).to(device)
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()
        dl_te = make_loader(args.data, "test", batch=args.batch, shuffle=False)
        acc = np.zeros(4); n = 0
        with torch.no_grad():
            for x, y in dl_te:
                x, y = x.to(device), y.to(device)
                acc += dice_per_class(model(x), y)
                n += 1
        dpc = acc / max(1, n)
        print("TEST Dice [BG, CSF, GM, WM] =>", dpc, "mean:", dpc.mean())
        raise SystemExit(0)

    dl_tr = make_loader(args.data, "train", batch=args.batch, shuffle=True)
    dl_va = make_loader(args.data, "val", batch=args.batch, shuffle=False)

    model = UNetImproved2D(in_ch=1, n_classes=4, base=args.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = DiceCELoss(n_classes=4)

    best = -1.0
    best_path = Path(args.out) / "best_unet.pth"

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()

        model.eval()
        acc = np.zeros(4); n = 0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(device), y.to(device)
                acc += dice_per_class(model(x), y)
                n += 1
        dpc = acc / max(1, n)
        mean_dice = dpc.mean()
        print(f"[{ep:02d}] loss {np.mean(losses):.4f} | val Dice BG {dpc[0]:.3f} CSF {dpc[1]:.3f} GM {dpc[2]:.3f} WM {dpc[3]:.3f} | mean {mean_dice:.3f} | {time.time()-t0:.1f}s")

        if mean_dice > best:
            best = mean_dice
            torch.save({"model": model.state_dict(), "dice": dpc, "epoch": ep}, str(best_path))
            print(f"  → saved {best_path}")
