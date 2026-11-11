"""Small shared helpers: image conversion, metrics, plotting."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def gray(img: np.ndarray) -> np.ndarray:
    """(H,W,C) -> (H,W) luminance; passes 2-D input through."""
    if img.ndim == 2:
        return img
    if img.shape[-1] == 1:
        return img[..., 0]
    return img[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=img.dtype)


def mse(a, b) -> float:
    return float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))


def psnr(a, b, peak: float = 1.0) -> float:
    m = mse(a, b)
    return float("inf") if m == 0 else 10 * np.log10(peak ** 2 / m)


def imshow(img, title: str = "", ax=None, cmap=None, **kw):
    """Show an image in [0,1] (or a scalar map with a colormap) with axes off."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]
    if img.ndim == 2 and cmap is None:
        ax.imshow(img, cmap="gray", **kw)
    else:
        ax.imshow(np.clip(img, 0, 1) if img.ndim == 3 else img, cmap=cmap, **kw)
    ax.set_title(title); ax.axis("off")
    return ax


def view_grid(lf, step: int = 1, figsize=(10, 10), title=None):
    """Plot every ``step``-th sub-aperture view as a grid (slide 10/11 figure)."""
    us = range(0, lf.nu, step); vs = range(0, lf.nv, step)
    fig, axes = plt.subplots(len(us), len(vs), figsize=figsize, squeeze=False)
    for i, u in enumerate(us):
        for j, v in enumerate(vs):
            axes[i, j].imshow(np.clip(lf.view(u, v), 0, 1)); axes[i, j].axis("off")
    fig.suptitle(title or f"{lf.name}: sub-aperture views (every {step})")
    plt.tight_layout()
    return fig


def epi_figure(lf, row: int | None = None, figsize=(12, 5)):
    """Central view with a marked row, and its horizontal EPI stretched vertically.
    Straight lines in the EPI have slope = disparity (Bolles, Baker & Marimont 1987)."""
    row = lf.height // 2 if row is None else row
    epi = lf.epi_horizontal(row)
    fig, (a, b) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [4, 1]})
    a.imshow(np.clip(lf.center_view, 0, 1)); a.axhline(row, color="r", lw=1); a.axis("off"); a.set_title("central view")
    b.imshow(np.clip(epi, 0, 1), aspect="auto", interpolation="nearest"); b.set_ylabel("v"); b.set_xlabel("t (column)")
    b.set_title(f"horizontal EPI at row {row}")
    plt.tight_layout()
    return fig
