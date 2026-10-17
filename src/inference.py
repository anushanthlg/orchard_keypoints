"""Run inference on an image or folder.

    python src/inference.py --images orchard_keypoint_images --out runs/preds
    python src/inference.py --images some.png
Outputs overlay images and predictions.json (coordinates in original image pixels).
The overlay also shows left_intercept->vanishing_point and
right_intercept->vanishing_point_right, extended to their intersection point.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import config as C
from model import KPNet
from utils import (decode_heatmaps, draw_convergence, draw_keypoints,
                   get_device, preprocess_bgr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(C.RUNS_DIR / "best.pt"))
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default=str(C.RUNS_DIR / "preds"))
    args = ap.parse_args()

    dev = get_device()
    model = KPNet(pretrained=False).to(dev)
    model.load_state_dict(torch.load(args.weights, map_location=dev)["model"])
    model.eval()

    src = Path(args.images)
    files = [src] if src.is_file() else sorted(
        p for p in src.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for f in files:
        bgr = cv2.imread(str(f))
        h, w = bgr.shape[:2]
        with torch.no_grad():
            pts, conf = decode_heatmaps(model(preprocess_bgr(bgr).to(dev)).cpu())
        pts = pts[0].numpy() * np.array([w / C.IMG_SIZE, h / C.IMG_SIZE])
        conf = conf[0].numpy()
        results[f.name] = {n: {"x": float(p[0]), "y": float(p[1]), "conf": float(c)}
                           for n, p, c in zip(C.KP_NAMES, pts, conf)}

        named = {n: (p[0], p[1]) for n, p in zip(C.KP_NAMES, pts)}
        draw_keypoints(bgr, named)
        X = draw_convergence(bgr, named)
        results[f.name]["intersection"] = {"x": X[0], "y": X[1]} if X else None
        cv2.imwrite(str(out / f.name), bgr)

    json.dump(results, open(out / "predictions.json", "w"), indent=2)
    print(f"{len(files)} images -> {out}")


if __name__ == "__main__":
    main()