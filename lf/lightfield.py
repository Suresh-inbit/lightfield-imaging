"""The LightField container used by every module in :mod:`lf`."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class LightField:
    """A 4-D light field ``L(u, v, s, t)`` stored as ``views[u, v, s, t, c]``.

    Parameters
    ----------
    views : ndarray, shape (U, V, H, W, C), float32 in [0, 1]
        Sub-aperture images on a regular angular grid.
    name : str
        Human-readable dataset name.
    meta : dict
        Free-form metadata (source URL, camera positions, ground-truth
        disparity, ...).  Modules may read but should not rely on it.
    """
    views: np.ndarray
    name: str = "lightfield"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        v = np.asarray(self.views)
        if v.ndim == 4:                       # grayscale without channel axis
            v = v[..., None]
        if v.ndim != 5:
            raise ValueError(f"views must be (U,V,H,W,C); got shape {v.shape}")
        if v.dtype != np.float32:
            v = v.astype(np.float32)
        self.views = v

    # ---- shape helpers -------------------------------------------------
    @property
    def nu(self) -> int: return self.views.shape[0]
    @property
    def nv(self) -> int: return self.views.shape[1]
    @property
    def height(self) -> int: return self.views.shape[2]
    @property
    def width(self) -> int: return self.views.shape[3]
    @property
    def channels(self) -> int: return self.views.shape[4]
    @property
    def shape(self): return self.views.shape
    @property
    def center(self) -> tuple[int, int]:
        """Index ``(u0, v0)`` of the central sub-aperture view."""
        return self.nu // 2, self.nv // 2

    # ---- accessors -----------------------------------------------------
    def view(self, u: int, v: int) -> np.ndarray:
        """Sub-aperture image at angular index ``(u, v)``, shape (H, W, C)."""
        return self.views[u, v]

    @property
    def center_view(self) -> np.ndarray:
        u0, v0 = self.center
        return self.views[u0, v0]

    def epi_horizontal(self, row: int, u: int | None = None) -> np.ndarray:
        """Horizontal epipolar-plane image: fix image row ``s=row`` and angular
        row ``u``; stack the row across all ``v``.  Shape (V, W, C).
        Scene points trace straight lines whose slope equals their disparity."""
        u = self.center[0] if u is None else u
        return self.views[u, :, row, :, :]

    def epi_vertical(self, col: int, v: int | None = None) -> np.ndarray:
        """Vertical EPI: fix image column ``t=col`` and angular column ``v``.
        Shape (U, H, C)."""
        v = self.center[1] if v is None else v
        return self.views[:, v, :, col, :]

    # ---- transforms (all return a new LightField) -----------------------
    def to_gray(self) -> "LightField":
        if self.channels == 1:
            return self
        w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        g = np.tensordot(self.views[..., :3], w, axes=([-1], [0]))[..., None]
        return LightField(g, self.name + "_gray", dict(self.meta))

    def crop_angular(self, n_u: int, n_v: int | None = None) -> "LightField":
        """Keep the central ``n_u x n_v`` views (digital stopping-down of the grid)."""
        n_v = n_u if n_v is None else n_v
        u0, v0 = self.center
        su, sv = u0 - n_u // 2, v0 - n_v // 2
        return LightField(self.views[su:su + n_u, sv:sv + n_v], self.name, dict(self.meta))

    def crop_spatial(self, r0: int, r1: int, c0: int, c1: int) -> "LightField":
        meta = dict(self.meta)
        if "gt_disparity" in meta and meta["gt_disparity"] is not None:
            meta["gt_disparity"] = meta["gt_disparity"][r0:r1, c0:c1]
        return LightField(self.views[:, :, r0:r1, c0:c1], self.name, meta)

    def downsample(self, factor: int) -> "LightField":
        """Spatial box-downsampling by an integer factor (fast, anti-aliased
        enough for experiments).  Disparities scale by ``1/factor``."""
        if factor == 1:
            return self
        U, V, H, W, C = self.shape
        H2, W2 = (H // factor) * factor, (W // factor) * factor
        v = self.views[:, :, :H2, :W2].reshape(U, V, H2 // factor, factor, W2 // factor, factor, C)
        out = v.mean(axis=(3, 5), dtype=np.float32)
        meta = dict(self.meta)
        if meta.get("gt_disparity") is not None:
            g = meta["gt_disparity"][:H2, :W2]
            meta["gt_disparity"] = g.reshape(H2 // factor, factor, W2 // factor, factor).mean(axis=(1, 3)) / factor
        if meta.get("disparity_range") is not None:
            lo, hi = meta["disparity_range"]
            meta["disparity_range"] = (lo / factor, hi / factor)
        return LightField(out, self.name, meta)

    def __repr__(self):
        U, V, H, W, C = self.shape
        return f"LightField('{self.name}', angular {U}x{V}, spatial {H}x{W}, {C} ch)"
