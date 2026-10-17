"""Heatmap encode/decode, drawing and misc helpers."""
import random

import cv2
import numpy as np
import torch

import config as C

COLORS = {
    "vanishing_point": (0, 255, 0),
    "vanishing_point_right": (0, 200, 255),
    "left_intercept": (255, 0, 255),
    "right_intercept": (255, 255, 0),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_heatmaps(pts, vis):
    """pts [4,2] in image pixels, vis [4] -> heatmaps [4,H,W]."""
    hm = np.zeros((len(C.KP_NAMES), C.HM_SIZE, C.HM_SIZE), np.float32)
    ys, xs = np.mgrid[0:C.HM_SIZE, 0:C.HM_SIZE]
    for k in range(len(C.KP_NAMES)):
        if vis[k]:
            cx, cy = pts[k] / C.STRIDE
            hm[k] = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * C.SIGMA ** 2))
    return hm


def decode_heatmaps(hm):
    """hm [B,K,H,W] tensor -> (pts [B,K,2] in image pixels, conf [B,K])."""
    B, K, H, W = hm.shape
    conf, idx = hm.reshape(B, K, -1).max(-1)
    ys, xs = (idx // W).float(), (idx % W).float()
    pts = torch.stack([xs, ys], -1)
    for b in range(B):                       # quarter-pixel refinement
        for k in range(K):
            x, y = int(xs[b, k]), int(ys[b, k])
            if 0 < x < W - 1:
                pts[b, k, 0] += 0.25 * torch.sign(hm[b, k, y, x + 1] - hm[b, k, y, x - 1])
            if 0 < y < H - 1:
                pts[b, k, 1] += 0.25 * torch.sign(hm[b, k, y + 1, x] - hm[b, k, y - 1, x])
    return (pts + 0.5) * C.STRIDE - 0.5, conf


def preprocess_bgr(bgr):
    """BGR uint8 image of any size -> normalized tensor [1,3,IMG,IMG]."""
    rgb = cv2.cvtColor(cv2.resize(bgr, (C.IMG_SIZE, C.IMG_SIZE)), cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0
    rgb = (rgb - np.array(C.MEAN, np.float32)) / np.array(C.STD, np.float32)
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy())[None]


def draw_keypoints(bgr, points, names=None, radius=4):
    """points: dict name -> (x, y) or list aligned with names."""
    if not isinstance(points, dict):
        points = dict(zip(names or C.KP_NAMES, points))
    for n, (x, y) in points.items():
        col = COLORS.get(n, (255, 255, 255))
        cv2.circle(bgr, (int(round(x)), int(round(y))), radius, col, -1)
        cv2.putText(bgr, n, (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
    return bgr


# ------------------------------------------------------------- geometry
def line_intersection(p1, p2, p3, p4):
    """Intersection of the infinite lines p1-p2 and p3-p4, or None if parallel."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-6:
        return None
    a = x1 * y2 - y1 * x2
    b = x3 * y4 - y3 * x4
    return ((a * (x3 - x4) - (x1 - x2) * b) / d,
            (a * (y3 - y4) - (y1 - y2) * b) / d)


def _dashed_line(img, p, q, color, dash=8):
    p, q = np.array(p, float), np.array(q, float)
    n = max(int(np.linalg.norm(q - p) // dash), 1)
    for i in range(0, n, 2):
        a = p + (q - p) * i / n
        b = p + (q - p) * min(i + 1, n) / n
        cv2.line(img, tuple(int(round(v)) for v in a),
                 tuple(int(round(v)) for v in b), color, 1)


def draw_convergence(bgr, pts):
    """pts: dict name -> (x, y).

    Draws left_intercept -> vanishing_point and right_intercept -> vanishing_point_right,
    extends both past the vanishing points (dashed) and marks where the two lines meet.
    Returns the intersection (x, y), or None if a keypoint is missing or the lines
    are (nearly) parallel.
    """
    need = ["left_intercept", "vanishing_point", "right_intercept", "vanishing_point_right"]
    if not all(n in pts for n in need):
        return None
    li, vl, ri, vr = (np.array(pts[n], float) for n in need)
    X = line_intersection(li, vl, ri, vr)

    h, w = bgr.shape[:2]
    ip = lambda p: tuple(int(round(v)) for v in p)
    cv2.line(bgr, ip(li), ip(vl), (0, 255, 255), 1)
    cv2.line(bgr, ip(ri), ip(vr), (0, 255, 255), 1)
    if X is None or max(abs(X[0]), abs(X[1])) > 1e4:
        return None
    X = np.array(X)

    # dashed extension only when the intersection lies beyond the vanishing point
    if np.dot(X - li, vl - li) > np.dot(vl - li, vl - li):
        _dashed_line(bgr, vl, X, (0, 255, 255))
    if np.dot(X - ri, vr - ri) > np.dot(vr - ri, vr - ri):
        _dashed_line(bgr, vr, X, (0, 255, 255))

    if 0 <= X[0] < w and 0 <= X[1] < h:
        cv2.drawMarker(bgr, ip(X), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
        cv2.putText(bgr, "intersection", (int(X[0]) + 6, int(X[1]) + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    return float(X[0]), float(X[1])