"""lf — a small, readable light-field imaging toolkit for EE 5176 experiments.

Conventions used everywhere in this package
-------------------------------------------
* A light field is stored as a NumPy array ``views`` of shape ``(U, V, H, W, C)``:
  ``U`` angular rows, ``V`` angular columns, ``H x W`` spatial pixels, ``C``
  colour channels (1 or 3).  Angular axes are oriented so that nearer scene
  points have positive disparity (see below).  Values are
  ``float32`` in ``[0, 1]`` (linear-ish intensity; the loaders do not undo gamma).
* ``views[u, v]`` is the *sub-aperture image* seen through aperture position
  ``(u, v)``.  ``(u0, v0) = (U//2, V//2)`` is the central view.
* Two-plane parameterisation (Ng 2005): ``L(u, v, s, t)`` with ``(u, v)`` on the
  lens/aperture plane and ``(s, t)`` on the sensor plane.  Our array index order
  is ``L[u, v, s, t]`` with ``s`` = image row (y) and ``t`` = image column (x).
* Disparity ``d`` of a scene point is the image shift (in pixels) per unit step
  of the camera in the angular grid.  Points on the plane the array was
  rectified to have ``d = 0``.
"""
from .lightfield import LightField
from . import io, synthetic, utils
# method modules (each is also importable on its own, e.g. ``from lf import refocus``)
from . import refocus, aperture, depth, edof, plenoptic, multiplex, hdr

__all__ = ["LightField", "io", "synthetic", "utils",
           "refocus", "aperture", "depth", "edof", "plenoptic", "multiplex", "hdr"]
