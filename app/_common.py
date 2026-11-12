"""Shared helpers for the Streamlit explorer: loading, caching, display, docs."""
from __future__ import annotations

import pathlib
import re
import sys
import traceback

import imageio.v3 as iio
import matplotlib
import numpy as np
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lf  # noqa: E402
from lf import refocus as rf  # noqa: E402

DOCS = ROOT / "docs"

# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading light field ...")
def load_lf(dataset: str, downsample: int, crop: int, size: int, n_views: int) -> lf.LightField:
    """Cached loader.  Stanford gantry data is never loaded below downsample=4."""
    if dataset == "synthetic":
        L = lf.io.load("synthetic", n_views=n_views, size=size)
    elif dataset.startswith("lytro"):
        L = lf.io.load(dataset, crop_angular=int(crop), downsample=int(downsample))
    else:                                   # any Stanford gantry scene on disk
        L = lf.io.load(dataset, downsample=max(int(downsample), 4), angular_crop=int(crop))
    return L


@st.cache_resource(show_spinner="Decoding uploaded light field ...", max_entries=2)
def load_uploaded(_data: bytes, digest: str, filename: str, angular: tuple, crop, downsample: int) -> lf.LightField:
    """Cached decode of an uploaded file (keyed on its SHA-1, not the bytes)."""
    return lf.io.load_upload(_data, filename, angular=tuple(angular), crop_angular=crop, downsample=int(downsample))


def alpha_range(dataset: str, downsample: int) -> tuple[float, float]:
    """Suggested refocus alpha range for the loaded light field.

    ``lf.refocus.SUGGESTED_ALPHA_RANGE`` is measured at a reference downsample
    (4 for the Stanford sets, 1 for lytro/synthetic); disparities scale with
    1/downsample, so the range is rescaled by ``reference / downsample``."""
    sug = getattr(rf, "SUGGESTED_ALPHA_RANGE", {}) or {}
    reference = {"lego": 4, "bracelet": 4, "lytro": 1, "synthetic": 1}
    if dataset in sug:
        lo, hi = sug[dataset]
        s = reference.get(dataset, 1) / max(float(downsample), 1.0)
        return float(lo * s), float(hi * s)
    if dataset not in ("upload",) and not dataset.startswith("lytro"):   # unmeasured Stanford scene
        s = 4.0 / max(float(downsample), 1.0)
        return -2.5 * s, 2.5 * s
    return -2.0, 2.0


# --------------------------------------------------------------------------
# display helpers
# --------------------------------------------------------------------------
def _img_u8(img):
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = lf.utils.to_uint8(a)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    return a


def show(img, caption=None, min_side: int = 0):
    """st.image for float [0,1] / uint8 arrays; optional integer up-scaling."""
    a = _img_u8(img)
    if min_side and min(a.shape[:2]) < min_side:
        f = int(np.ceil(min_side / min(a.shape[:2])))
        a = np.repeat(np.repeat(a, f, axis=0), f, axis=1)
    st.image(a, caption=caption, clamp=True, width="stretch")


def show_fig(fig):
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def gif_bytes(frames, ms: int = 90) -> bytes:
    arr = np.stack([_img_u8(f) for f in frames])
    return iio.imwrite("<bytes>", arr, extension=".gif", duration=ms, loop=0)


def mosaic(tiles, ncols: int, gap: int = 2, fill: float = 0.5):
    """Tile equally-sized (h, w[, c]) arrays into one image."""
    tiles = [np.asarray(t, np.float32) for t in tiles]
    if tiles[0].ndim == 2:
        tiles = [t[..., None] for t in tiles]
    h, w, c = tiles[0].shape
    nrows = int(np.ceil(len(tiles) / ncols))
    out = np.full((nrows * (h + gap) - gap, ncols * (w + gap) - gap, c), fill, np.float32)
    for i, t in enumerate(tiles):
        r, q = divmod(i, ncols)
        out[r * (h + gap):r * (h + gap) + h, q * (w + gap):q * (w + gap) + w] = t
    return out[..., 0] if c == 1 else out


def box(img_u8, r0, r1, c0, c1, color=(255, 40, 40)):
    """Draw a rectangle outline on a uint8 image copy."""
    a = np.array(_img_u8(img_u8))
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    t = max(1, min(a.shape[:2]) // 128)
    r0, r1, c0, c1 = int(r0), int(r1), int(c0), int(c1)
    for sl in (a[r0:r0 + t, c0:c1], a[max(r1 - t, 0):r1, c0:c1],
               a[r0:r1, c0:c0 + t], a[r0:r1, max(c1 - t, 0):c1]):
        sl[...] = color
    return a


# --------------------------------------------------------------------------
# run-on-demand gate for heavy tabs
# --------------------------------------------------------------------------
def gate(key: str, label: str, params: dict, lkey: str, help: str | None = None):
    """Button that freezes ``params`` for a heavy computation.  Returns the frozen
    parameter dict (computed results come from st.cache_data) or None."""
    if st.button(label, key=f"{key}_btn|{lkey}", help=help, type="primary"):
        st.session_state[key] = (lkey, dict(params))
    got = st.session_state.get(key)
    if got is None or got[0] != lkey:
        st.caption("Press the button to compute (results are cached).")
        return None
    if got[1] != params:
        st.caption("Controls changed since the last run; press the button again to update.")
    return got[1]


def safe(name: str, fn, *args):
    """Run a tab body; one failing tab must not kill the app."""
    try:
        fn(*args)
    except Exception as e:  # noqa: BLE001
        st.error(f"The '{name}' tab failed: {e}")
        st.exception(e)


# --------------------------------------------------------------------------
# documentation rendering
# --------------------------------------------------------------------------
_IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
_REF_RE = re.compile(r"`(?:docs/)?figures/([\w\-.]+\.(?:png|gif|jpg))`")


def _resolve(href: str):
    for p in (DOCS / href, ROOT / href, DOCS / "figures" / pathlib.Path(href).name):
        if p.exists():
            return p
    return None


def render_note(filename: str):
    """Render docs/<filename> with st.markdown; figures shown with st.image."""
    path = DOCS / filename
    if not path.exists():
        st.warning(f"{path} not found")
        return
    text = path.read_text(encoding="utf-8")
    shown, pos = set(), 0
    for m in _IMG_RE.finditer(text):
        chunk = text[pos:m.start()]
        if chunk.strip():
            st.markdown(chunk)
        p = _resolve(m.group(2))
        if p is not None:
            st.image(str(p), caption=m.group(1), width="stretch")
            shown.add(p.name)
        pos = m.end()
    if text[pos:].strip():
        st.markdown(text[pos:])
    extra = []
    for n in dict.fromkeys(_REF_RE.findall(text)):
        p = DOCS / "figures" / n
        if n not in shown and p.exists():
            extra.append(p)
    if extra:
        st.markdown("**Figures referenced in this note**")
        cols = st.columns(2)
        for i, p in enumerate(extra):
            with cols[i % 2]:
                st.image(str(p), caption=p.name, width="stretch")


def explainer(text: str, note: str):
    with st.expander("How this works"):
        st.markdown(text)
    with st.expander("Full note"):
        render_note(note)
