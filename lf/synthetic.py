"""A synthetic light field with known geometry, for tests and for explaining
algorithms before touching real data.

The scene is a stack of fronto-parallel textured layers.  A layer at disparity
``d`` appears shifted by ``d * (v - v0)`` pixels horizontally and ``d * (u - u0)``
vertically in view ``(u, v)`` — exactly the geometry of a camera translating on
a plane (two-plane parameterisation).  Layers are composited back-to-front so
occlusions are physically consistent.  The ground-truth disparity map of the
central view is stored in ``meta['gt_disparity']``.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import shift as nd_shift
from .lightfield import LightField


def _texture(kind: str, h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Return an (h, w, 3) float texture in [0,1] built from skimage.data."""
    import skimage.data as skd
    from skimage.transform import resize
    if kind == "grass":
        t = skd.grass()
    elif kind == "brick":
        t = skd.brick()
    elif kind == "gravel":
        t = skd.gravel()
    elif kind == "text":
        t = skd.text()
    elif kind == "camera":
        t = skd.camera()
    else:
        t = rng.random((h, w))
    t = t.astype(np.float32)
    t = (t - t.min()) / (np.ptp(t) + 1e-8)
    t = resize(t, (h, w), anti_aliasing=True).astype(np.float32)
    tint = np.array({"grass": [0.55, 0.9, 0.45], "brick": [0.95, 0.55, 0.4],
                     "gravel": [0.8, 0.8, 0.85], "text": [0.9, 0.9, 0.6],
                     "camera": [0.7, 0.8, 1.0]}.get(kind, [1, 1, 1]), dtype=np.float32)
    return np.clip(0.15 + 0.85 * t[..., None] * tint, 0, 1)


def synthetic_light_field(n_views: int = 9, size: int = 256, seed: int = 0,
                          layers=None, name: str = "synthetic") -> LightField:
    """Build a ``n_views x n_views`` light field of ``size x size`` pixels.

    Parameters
    ----------
    layers : list of dicts ``{'disparity': d, 'texture': str, 'mask': (kind, params)}``
        Front-most last.  Default: grass background (d=-1.0), a brick square
        (d=0.0, in focus at the rectification plane) and a gravel disc (d=+1.5,
        nearest).  Disparities are in pixels per angular step.
    """
    rng = np.random.default_rng(seed)
    if layers is None:
        layers = [
            {"disparity": -1.0, "texture": "grass", "mask": ("full",)},
            {"disparity": 0.0, "texture": "text", "mask": ("rect", 0.10, 0.10, 0.55, 0.60)},
            {"disparity": 1.5, "texture": "gravel", "mask": ("disc", 0.68, 0.66, 0.20)},
        ]
    H = W = size
    yy, xx = np.mgrid[0:H, 0:W] / float(size)
    # pad textures so shifted layers never show empty borders
    pad = int(np.ceil(max(abs(l["disparity"]) for l in layers) * n_views)) + 2
    texs, masks = [], []
    for l in layers:
        t = _texture(l["texture"], H + 2 * pad, W + 2 * pad, rng)
        kind = l["mask"][0]
        if kind == "full":
            m = np.ones((H, W), np.float32)
        elif kind == "rect":
            _, y0, x0, y1, x1 = l["mask"]
            m = ((yy >= y0) & (yy < y1) & (xx >= x0) & (xx < x1)).astype(np.float32)
        elif kind == "disc":
            _, cy, cx, r = l["mask"]
            m = (((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r).astype(np.float32)
        else:
            raise ValueError(kind)
        mp = np.zeros((H + 2 * pad, W + 2 * pad), np.float32); mp[pad:pad + H, pad:pad + W] = m
        texs.append(t); masks.append(mp)

    u0 = v0 = n_views // 2
    views = np.zeros((n_views, n_views, H, W, 3), np.float32)
    gt = np.zeros((H, W), np.float32)
    for l, m in zip(layers, masks):               # back-to-front for gt
        mc = m[pad:pad + H, pad:pad + W]
        gt = np.where(mc > 0.5, l["disparity"], gt)
    for u in range(n_views):
        for v in range(n_views):
            img = np.zeros((H, W, 3), np.float32)
            for l, t, m in zip(layers, texs, masks):
                dy, dx = l["disparity"] * (u - u0), l["disparity"] * (v - v0)
                ts = nd_shift(t, (dy, dx, 0), order=1, mode="nearest")[pad:pad + H, pad:pad + W]
                ms = nd_shift(m, (dy, dx), order=1, mode="constant")[pad:pad + H, pad:pad + W][..., None]
                img = img * (1 - ms) + ts * ms
            views[u, v] = img
    meta = {"source": "synthetic (lf.synthetic)", "gt_disparity": gt,
            "disparity_range": (min(l["disparity"] for l in layers), max(l["disparity"] for l in layers)),
            "layers": layers}
    return LightField(views, name, meta)
