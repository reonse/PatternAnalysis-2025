import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from modules import UNetImproved2D

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--out", type=str, default="outputs/prediction.png")
    ap.add_argument("--base", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = Image.open(args.image).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

    model = UNetImproved2D(in_ch=1, n_classes=4, base=args.base).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        logits = model(x)
        pred = F.softmax(logits, dim=1).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    out = Image.fromarray(pred)
    out.save(args.out)
    print(f"Saved: {args.out}")
