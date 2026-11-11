"""Interactive light-field explorer (EE 5176 Computational Photography).

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from _common import (ROOT, alpha_range, box, explainer, gate, gif_bytes, load_lf,  # noqa: E402,F401
                     mosaic, plt, safe, show, show_fig)
import lf  # noqa: E402
from lf import aperture, depth, edof, hdr, multiplex, plenoptic, refocus  # noqa: E402

st.set_page_config(page_title="Light-field explorer", layout="wide")

# ==========================================================================
# cached computations (the LightField arg is excluded from hashing; ``lk`` keys it)
# ==========================================================================
@st.cache_data(show_spinner=False, max_entries=64)
def c_refocus(_L, lk, alpha, method, radius, shape, full):
    if full:
        return refocus.refocus(_L, alpha, method=method)
    return aperture.stop_down(_L, radius, alpha=alpha, shape=shape, method=method)


@st.cache_data(show_spinner="Building focal stack ...", max_entries=8)
def c_stack(_L, lk, lo, hi, n, method, radius, shape, full):
    # same aperture and method as the main image, so the strip matches it
    return np.stack([c_refocus(_L, lk, float(a), method, radius, shape, full)
                     for a in np.linspace(lo, hi, n)])


@st.cache_data(show_spinner="Sweeping sharpness ...", max_entries=8)
def c_sweep(_L, lk, lo, hi, n, r0, r1, c0, c1, method, radius, shape, full, alpha_now):
    """Sharpness curve computed with exactly the settings of the live readout
    (method + aperture).  The current alpha is added to the sample grid so the
    curve passes through the number shown in the metric."""
    al = np.unique(np.append(np.linspace(lo, hi, n), alpha_now))
    mask = np.zeros((_L.height, _L.width), bool); mask[r0:r1, c0:c1] = True
    s = np.array([refocus.sharpness(c_refocus(_L, lk, float(a), method, radius, shape, full), mask) for a in al])
    return al, s


@st.cache_data(show_spinner="Rendering sweep GIF ...", max_entries=4)
def c_sweep_gif(_L, lk, lo, hi, n, method):
    al = np.concatenate([np.linspace(lo, hi, n), np.linspace(hi, lo, n)[1:-1]])
    return gif_bytes([refocus.refocus(_L, float(a), method=method) for a in al], 80)


@st.cache_data(show_spinner="Rendering perspective path ...", max_entries=4)
def c_persp_gif(_L, lk, path, n, radius):
    return gif_bytes(aperture.perspective_sequence(_L, path=path, n=n, radius=radius), 80)


@st.cache_resource(show_spinner=False)
def c_lenslet(_L, lk):
    return plenoptic.to_lenslet(_L)


@st.cache_data(show_spinner="Estimating depth ...", max_entries=8)
def c_depth(_L, lk, est, lo, hi, n, window):
    al = np.linspace(lo, hi, n)
    if est == "defocus":
        r = depth.depth_from_focus(_L, al, window=window)
    elif est == "correspondence":
        r = depth.depth_from_correspondence(_L, al, window=max(3, window - 2))
    elif est == "EPI structure tensor":
        r = depth.epi_structure_tensor_disparity(_L)
    else:
        r = depth.depth_combined(_L, al, window_focus=window, window_corr=max(3, window - 2))
    return {k: np.asarray(r[k]) for k in ("disparity", "confidence")}


@st.cache_data(show_spinner="Building all-in-focus image ...", max_entries=8)
def c_aif(_L, lk, lo, hi, n, window):
    r = edof.all_in_focus(_L, np.linspace(lo, hi, n), window=window)
    return {"image": r["image"], "disparity": r["disparity"]}


@st.cache_data(show_spinner="Computing slide-58 comparison ...", max_entries=8)
def c_triptych(_L, lk, lo, hi, n, window, radius, noise, focus_alpha):
    return edof.compare_with_stop_down(_L, np.linspace(lo, hi, n), small_radius=radius,
                                       noise_sigma=noise or None, focus_alpha=focus_alpha,
                                       window=window)


@st.cache_data(show_spinner=False, max_entries=32)
def c_patterns(kind, nu, nv, iters, seed):
    n = nu * nv
    if kind == "identity":
        return multiplex.identity_patterns(n)
    if kind == "S-matrix" and n in multiplex.valid_s_sizes(64):
        return multiplex.s_matrix(n)
    if kind == "random":
        return multiplex.random_patterns(n, 0.5, rng=seed)
    return multiplex.optimize_patterns(n, iters=iters, noise="read", rng=seed)[0]


@st.cache_data(show_spinner="Simulating captures ...", max_entries=16)
def c_multiplex(_L, lk, nu, nv, kind, iters, seed, T, photon, read):
    Lc = _L.crop_angular(nu, nv)
    out = {}
    for k in ("identity", "mux", "random"):
        kk = {"mux": kind if kind not in ("identity", "random") else "optimised"}.get(k, k)
        W = c_patterns(kk if k == "mux" else k, nu, nv, iters, seed)
        Y = multiplex.capture(Lc, W, T, photon, read, rng=seed)
        X = multiplex.demultiplex(Y, W, T / W.shape[0], angular=(nu, nv), clip=True)
        out[k] = {"W": W, "shot": Y[0] / max(T / W.shape[0] * W[0].sum(), 1e-9),
                  "views": X.views, "psnr": lf.utils.psnr(X.views, Lc.views),
                  "psnr_c": lf.utils.psnr(X.views[Lc.center], Lc.center_view)}
    return out


@st.cache_resource(show_spinner="Building HDR light field ...")
def c_hdr_source(source, lk, n_views, size, _L):
    if source == "synthetic HDR scene":
        return hdr.synthetic_hdr_light_field(n_views=n_views, size=size)
    Ls = _L.crop_angular(min(5, _L.nu, _L.nv))
    if Ls.height > 128:
        Ls = Ls.downsample(int(np.ceil(Ls.height / 128)))
    return hdr.hdr_from_ldr_light_field(Ls)


@st.cache_data(show_spinner="Capturing and merging HDR ...", max_entries=8)
def c_hdr_run(_R, rk, strategy, kind, exps, alpha, crf_photon):
    sim = dict(photon_scale=crf_photon)
    if strategy.startswith("A (gt") or strategy.startswith("A (no"):
        r = hdr.strategy_a(_R, kind, "gt" if "gt" in strategy else None, exps, **sim)
        return {"lf": r["hdr_lf"].views, "ldr": r["ldr_lf"].views, "em": r["exposure_map"],
                "log_rmse": r["log_rmse"], "shots": 1}
    if strategy.startswith("A (refocused"):
        em = hdr.exposure_pattern(_R.nu, _R.nv, kind, exps)
        ldr, em = hdr.capture_bracketed_views(_R, em, **sim)
        img = hdr.merge_bracketed_refocused(ldr, em, alpha)
        gt = refocus.refocus(_R, alpha)
        return {"image": img, "ldr": ldr.views, "em": em, "log_rmse": hdr.log_rmse(img, gt), "shots": 1}
    if strategy.startswith("C"):
        r = hdr.temporal_bracket_hdr_lf(_R, exps, **sim)
        return {"lf": r["hdr_lf"].views, "ldr_c": np.stack([l.center_view for l in r["ldr_lfs"]]),
                "exps": np.array(exps), "log_rmse": r["log_rmse"], "shots": r["shots"]}
    r = hdr.aperture_area_bracketing(_R, alpha=alpha, **sim)
    return {"image": r["image"], "ldr_d": np.stack(r["ldr"]), "exps": r["exposures"],
            "log_rmse": r["log_rmse"], "shots": r["shots"]}


@st.cache_data(show_spinner="Comparing LDR vs HDR refocus ...", max_entries=8)
def c_hdr_cmp(_R, rk, alpha, exposure, est_views, photon):
    hl = lf.LightField(est_views) if est_views is not None else None
    r = hdr.ldr_refocus_comparison(_R, alpha, exposure, hdr_lf=hl, photon_scale=photon)
    return {k: v for k, v in r.items()}


# ==========================================================================
# sidebar
# ==========================================================================
st.sidebar.title("Light-field explorer")
avail = [k for k, v in lf.io.available().items() if v]
order = [k for k in ("synthetic", "lego", "bracelet", "lytro") if k in avail]
dataset = st.sidebar.selectbox("Dataset", order, key="dataset")

downsample, crop, size, nviews = 1, 9, 128, 7
if dataset in ("lego", "bracelet"):
    downsample = st.sidebar.selectbox("Spatial downsample", [4, 8, 16], 0, key="ds_stanford",
                                      help="Stanford data is never loaded below 4 (1024x1024x289 is 3.6 GB).")
    crop = st.sidebar.selectbox("Angular crop (central n x n)", [3, 5, 7, 9, 11, 13], 3, key="crop_st")
elif dataset == "lytro":
    downsample = st.sidebar.selectbox("Spatial downsample", [1, 2, 4], 1, key="ds_lytro")
    crop = st.sidebar.selectbox("Angular crop (central n x n)", [5, 7, 9, 11, 14], 2, key="crop_ly")
else:
    size = st.sidebar.select_slider("Synthetic size (px)", [64, 96, 128, 192, 256], 128, key="syn_size")
    nviews = st.sidebar.select_slider("Synthetic views per axis", [3, 5, 7, 9, 11], 7, key="syn_views")

try:
    L = load_lf(dataset, downsample, crop, size, nviews)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load {dataset}: {e}")
    st.exception(e)
    st.stop()

LK = f"{dataset}|{downsample}|{crop}|{size}|{nviews}"
A_LO, A_HI = alpha_range(dataset, downsample)
GT = L.meta.get("gt_disparity")

st.sidebar.markdown("**Light field**")
st.sidebar.markdown(f"angular **{L.nu} x {L.nv}**  \nspatial **{L.height} x {L.width}**, {L.channels} ch  \n"
                    f"memory **{L.views.nbytes / 1e6:.0f} MB**  \nsuggested alpha **{A_LO:.2f} .. {A_HI:.2f}**")
if dataset != "synthetic":
    st.sidebar.caption(f"Disparity units scale as 1/downsample (currently {downsample}).")


def K(name):  # widget keys depend on the loaded LF so state never outlives its range
    return f"{name}|{LK}"


st.title("Light-field imaging toolkit")
st.caption("EE 5176 Computational Photography - lecture 15 (Ng et al. 2005) and lecture 16 (Liang et al. 2008)")

# ==========================================================================
# 1. Explore
# ==========================================================================
def tab_explore():
    explainer(
        "A light field is a stack of pictures taken from a grid of viewpoints on the lens. "
        "Neighbouring views differ by parallax: a point with disparity $d$ moves $d$ pixels per angular step. "
        "If you fix one image row and stack that row across all views you get an epipolar-plane image (EPI), "
        "in which every scene point is a straight line of slope $d$. Nearer points make steeper lines, "
        "and occluders cut through the lines of the things behind them. "
        "Use the slope overlays to measure the disparity of a feature by eye.", "00_background.md")
    c1, c2 = st.columns([3, 2])
    with c1:
        step = st.slider("Show every k-th view", 1, max(1, min(L.nu, L.nv) - 1) if min(L.nu, L.nv) > 1 else 1,
                         2 if min(L.nu, L.nv) > 4 else 1, key=K("ex_step"))
        us, vs = range(0, L.nu, step), range(0, L.nv, step)
        show(mosaic([L.views[u, v] for u in us for v in vs], len(vs)),
             f"views u in {list(us)}, v in {list(vs)}")
    with c2:
        u = st.slider("view u (row)", 0, L.nu - 1, L.center[0], key=K("ex_u"))
        v = st.slider("view v (column)", 0, L.nv - 1, L.center[1], key=K("ex_v"))
        show(L.view(u, v), f"view ({u}, {v})")
    st.markdown("**Epipolar-plane images with slope lines** (line: $t = t_0 + d\\,(v - v_0)$)")
    d1, d2, d3 = st.columns(3)
    ds = [d1.number_input("disparity d1", value=float(round(A_LO, 2)), step=0.1, key=K("ex_d1")),
          d2.number_input("disparity d2", value=0.0, step=0.1, key=K("ex_d2")),
          d3.number_input("disparity d3", value=float(round(A_HI, 2)), step=0.1, key=K("ex_d3"))]
    colors = ["tab:red", "yellow", "tab:cyan"]
    r1, r2 = st.columns(2)
    row = r1.slider("image row s (horizontal EPI)", 0, L.height - 1, L.height // 2, key=K("ex_row"))
    col = r2.slider("image column t (vertical EPI)", 0, L.width - 1, L.width // 2, key=K("ex_col"))
    a1, a2 = st.columns(2)
    t0 = a1.slider("anchor column t0 of the lines", 0, L.width - 1, L.width // 2, key=K("ex_t0"))
    s0 = a2.slider("anchor row s0 of the lines", 0, L.height - 1, L.height // 2, key=K("ex_s0"))
    u0, v0 = L.center
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    eh = L.epi_horizontal(row, u)
    axs[0].imshow(np.clip(eh, 0, 1).squeeze(), aspect="auto", interpolation="nearest")
    for d, c in zip(ds, colors):
        axs[0].plot([t0 + d * (0 - v0), t0 + d * (L.nv - 1 - v0)], [0, L.nv - 1], c=c, lw=1.5, label=f"d={d:g}")
    axs[0].set(xlim=(-0.5, L.width - 0.5), ylim=(L.nv - 0.5, -0.5), xlabel="t (column)", ylabel="v",
               title=f"horizontal EPI, row {row}, u={u}")
    axs[0].legend(loc="upper right", fontsize=7)
    ev = L.epi_vertical(col, v)
    axs[1].imshow(np.clip(ev, 0, 1).squeeze(), aspect="auto", interpolation="nearest")
    for d, c in zip(ds, colors):
        axs[1].plot([s0 + d * (0 - u0), s0 + d * (L.nu - 1 - u0)], [0, L.nu - 1], c=c, lw=1.5)
    axs[1].set(xlim=(-0.5, L.height - 0.5), ylim=(L.nu - 0.5, -0.5), xlabel="s (row)", ylabel="u",
               title=f"vertical EPI, column {col}, v={v}")
    fig.tight_layout()
    show_fig(fig)
    show(box(L.view(u, v), row - 1, row + 2, 0, L.width), "horizontal EPI row (red) in the selected view")


# ==========================================================================
# 2. Plenoptic camera
# ==========================================================================
def tab_plenoptic():
    explainer(
        "A plenoptic camera puts a micro-lens array on the sensor; each micro-lens covers a $U\\times V$ block "
        "of pixels, and pixel $(i,j)$ of the block records the ray that passed through aperture position $(u,v)$. "
        "The raw image is therefore the light field re-interleaved into tiles. "
        "Taking pixel $(i,j)$ from under every micro-lens rebuilds one sub-aperture view, "
        "and averaging all pixels of a block gives the photo a normal camera would take. "
        "The price is resolution: the sensor's pixels are shared between $U\\cdot V$ views, "
        "so the spatial resolution is $M/N$.", "02_plenoptic_camera.md")
    raw = c_lenslet(L, LK)
    U, V = L.nu, L.nv
    st.subheader("Simulated raw lenslet image and zoom")
    a, b, c = st.columns(3)
    nl = a.slider("micro-lenses in zoom window", 3, 16, 6, key=K("pl_n"))
    zs = b.slider("zoom centre s (micro-lens row)", 0, L.height - 1, L.height // 2, key=K("pl_s"))
    zt = c.slider("zoom centre t (micro-lens column)", 0, L.width - 1, L.width // 2, key=K("pl_t"))
    cl, cr = st.columns([1, 1])
    with cl:
        small = raw[::max(1, raw.shape[0] // 700), ::max(1, raw.shape[1] // 700)]
        show(small, f"raw sensor mosaic {raw.shape[0]} x {raw.shape[1]} (subsampled for display)")
    with cr:
        show(plenoptic.lenslet_closeup(raw, (U, V), (zs, zt), nl), f"{nl} x {nl} micro-lens close-up", min_side=420)

    st.subheader("Sub-aperture extraction")
    a, b = st.columns(2)
    pi = a.slider("pixel row i under each micro-lens", 0, U - 1, U // 2, key=K("pl_i"))
    pj = b.slider("pixel column j under each micro-lens", 0, V - 1, V // 2, key=K("pl_j"))
    n = min(6, L.height, L.width)
    s0 = int(np.clip(zs - n // 2, 0, max(L.height - n, 0)))
    t0 = int(np.clip(zt - n // 2, 0, max(L.width - n, 0)))
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(np.clip(raw[s0 * U:(s0 + n) * U, t0 * V:(t0 + n) * V], 0, 1).squeeze(), interpolation="nearest")
    for ss in range(n):
        for tt in range(n):
            axs[0].add_patch(plt.Rectangle((tt * V + pj - 0.5, ss * U + pi - 0.5), 1, 1, fill=False, ec="red", lw=1.5))
    axs[0].set_title(f"pixel ({pi},{pj}) under every micro-lens")
    axs[1].imshow(np.clip(L.views[pi, pj], 0, 1).squeeze())
    axs[1].add_patch(plt.Rectangle((t0 - 0.5, s0 - 0.5), n, n, fill=False, ec="red", lw=1.5))
    axs[1].set_title(f"resulting view (u,v)=({pi},{pj})")
    axs[2].imshow(np.clip(L.views[pi, pj, s0:s0 + n, t0:t0 + n], 0, 1).squeeze(), interpolation="nearest")
    axs[2].set_title("window in that view = highlighted pixels")
    for ax in axs:
        ax.axis("off")
    fig.tight_layout()
    show_fig(fig)

    st.subheader("Conventional photo vs digitally stopped-down photo")
    rmax = float(max(U, V) // 2)
    rad = st.slider("aperture radius (pixels of the U x V block)", 0.0, rmax, min(1.0, rmax), 0.5, key=K("pl_rad"))
    cols = st.columns(3)
    with cols[0]:
        show(plenoptic.conventional_photo(L), "all pixels (full aperture, small DoF)")
    with cols[1]:
        show(plenoptic.conventional_photo(L, radius=rad), f"disc radius {rad:g} (stopped down)")
    with cols[2]:
        show(plenoptic.conventional_photo(L, radius=0.0), "single pixel (pinhole, max DoF)")

    st.subheader("Optical design calculator (slides 31-37)")
    g = st.columns(3)
    sh = g[0].number_input("sensor rows (px)", 100, 20000, 4000, 100, key="pl_sh")
    sw = g[1].number_input("sensor columns (px)", 100, 20000, 4000, 100, key="pl_sw")
    pp = g[2].number_input("pixel pitch (um)", 0.5, 50.0, 9.0, 0.5, key="pl_pp")
    g = st.columns(3)
    mp = g[0].number_input("micro-lens pitch (um)", 5.0, 500.0, 125.0, 5.0, key="pl_mp")
    mf = g[1].number_input("micro-lens focal length (um)", 10.0, 2000.0, 500.0, 10.0, key="pl_mf")
    fn = g[2].number_input("main-lens f-number", 0.7, 32.0, 4.0, 0.1, key="pl_fn")
    D = plenoptic.design((int(sh), int(sw)), pp, mp, mf, fn)
    m = st.columns(4)
    m[0].metric("angular resolution", "%d x %d" % D["angular_resolution"])
    m[1].metric("spatial resolution", "%d x %d" % D["spatial_resolution"])
    m[2].metric("micro-lens f/#", f"{D['microlens_f_number']:.2f}")
    m[3].metric("total rays (M)", f"{D['total_rays'] / 1e6:.1f}")
    m = st.columns(4)
    m[0].metric("aperture image diameter", f"{D['microlens_image_diameter_px']:.1f} px")
    m[1].metric("overlap with neighbours", f"{100 * D['overlap_fraction']:.0f} %")
    m[2].metric("wasted cell area", f"{100 * D['wasted_fraction']:.0f} %")
    m[3].metric("f-numbers match", "yes" if D["f_numbers_match"] else "NO")
    if not D["f_numbers_match"]:
        st.warning("Main-lens f-number differs from the micro-lens f-number: apertures either leave gaps "
                   "(main lens too slow) or overlap their neighbours (too fast).")
    st.markdown("**Resolution trade-off table (lecture 16, slide 14)** - spatial = sensor / angular")
    rows = [{"camera": r["camera"], "sensor (px)": "%d x %d" % r["sensor_px"], "angular": "%d x %d" % r["angular"],
             "spatial (slide)": "%d x %d" % r["spatial_slide"], "spatial (M/N)": "%d x %d" % r["spatial_computed"]}
            for r in plenoptic.camera_table()]
    ang = D["angular_resolution"]
    rows.append({"camera": "your design", "sensor (px)": f"{int(sh)} x {int(sw)}", "angular": "%d x %d" % ang,
                 "spatial (slide)": "-", "spatial (M/N)": "%d x %d" % plenoptic.resolution_tradeoff((int(sh), int(sw)), ang)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


# ==========================================================================
# 3. Refocus
# ==========================================================================
def tab_refocus():
    explainer(
        "To focus after the fact, shift each sub-aperture view in proportion to its angular offset from the centre and "
        "average: $I_\\alpha(s,t)=\\frac1N\\sum_{u,v}L(u,v,s+\\alpha(u-u_0),t+\\alpha(v-v_0))$. "
        "Points whose disparity equals $\\alpha$ line up and stay sharp, everything else is averaged over a patch "
        "that grows with $|d-\\alpha|$. Using only a disc of views is a smaller aperture: more depth of field, less light. "
        "The sharpness readout (variance of the Laplacian) peaks when your ROI is in focus.", "03_refocusing.md")
    c1, c2 = st.columns([2, 1])
    with c2:
        alpha = st.slider("alpha (refocus disparity)", float(round(A_LO - 0.5, 2)), float(round(A_HI + 0.5, 2)),
                          float(np.clip(0.0, A_LO, A_HI)), 0.05, key=K("rf_alpha"))
        method = st.selectbox("method", ["subpixel", "integer", "fourier"], key=K("rf_method"))
        shape = st.selectbox("aperture shape", ["circle", "square", "annulus"], key=K("rf_shape"))
        full_r = max(L.nu, L.nv) / 2.0
        radius = st.slider("aperture radius (views)", 0.0, float(full_r), float(full_r), 0.5, key=K("rf_rad"))
        rr = st.slider("ROI rows", 0, L.height, (L.height // 3, 2 * L.height // 3), key=K("rf_rr"))
        cc = st.slider("ROI columns", 0, L.width, (L.width // 3, 2 * L.width // 3), key=K("rf_cc"))
    full = radius >= full_r - 1e-6 and shape != "annulus"
    with st.spinner("Refocusing ..."):
        img = c_refocus(L, LK, alpha, method, radius, shape, full)
    mask = np.zeros(img.shape[:2], bool)
    mask[rr[0]:rr[1], cc[0]:cc[1]] = True
    sharp = refocus.sharpness(img, mask)
    with c1:
        show(box(lf.utils.to_uint8(img), rr[0], rr[1], cc[0], cc[1]),
             f"refocused at alpha={alpha:.2f} ({method}, {'full aperture' if full else f'{shape} r={radius:g}'})")
    with c2:
        st.metric("ROI sharpness (Laplacian variance)", f"{sharp:.3g}")
    st.markdown("**Focal stack and sharpness curve**")
    n = st.slider("images in stack", 3, 9, 5, key=K("rf_n"))
    run = gate("rf_stack", "Show focal-stack strip",
               dict(n=n, method=method, rr=rr, cc=cc, radius=radius, shape=shape, full=full), LK)
    if run:
        lo, hi = float(A_LO), float(A_HI)
        stack = c_stack(L, LK, lo, hi, run["n"], run["method"], run["radius"], run["shape"], run["full"])
        als = np.linspace(lo, hi, run["n"])
        per = 5
        for r0 in range(0, run["n"], per):
            cols = st.columns(per)
            for i, col in zip(range(r0, min(r0 + per, run["n"])), cols):
                with col:
                    show(stack[i], f"alpha={als[i]:.2f}")
        a, s = c_sweep(L, LK, lo, hi, 29, run["rr"][0], run["rr"][1], run["cc"][0], run["cc"][1],
                       run["method"], run["radius"], run["shape"], run["full"], float(alpha))
        fig, ax = plt.subplots(figsize=(6, 2.6))
        ax.plot(a, s, "o-")
        ax.axvline(alpha, c="r", ls="--", lw=1)
        ax.set(xlabel="alpha", ylabel="ROI sharpness", title="sharpness vs alpha in the ROI (red: current alpha)")
        ax.plot([alpha], [sharp], "rs", ms=6)
        fig.tight_layout()
        show_fig(fig)
        st.caption(f"ROI is sharpest near alpha = {a[int(np.argmax(s))]:.2f} "
                   f"(same method and aperture as the image above). With the 'subpixel' method, "
                   "whole-number alphas skip interpolation and read slightly sharper than fractional "
                   "ones; use 'fourier' for an unbiased comparison.")
    st.markdown("**Animate sweep**")
    ng = st.slider("frames per sweep direction", 6, 24, 12, key=K("rf_ng"))
    if st.button("Animate sweep (GIF)", key=K("rf_anim")):
        st.session_state["rf_gif"] = (LK, c_sweep_gif(L, LK, float(A_LO), float(A_HI), ng, method))
    g = st.session_state.get("rf_gif")
    if g and g[0] == LK:
        st.image(g[1], caption="refocus sweep", width="stretch")


# ==========================================================================
# 4. Perspective
# ==========================================================================
def tab_perspective():
    explainer(
        "Each sub-aperture image is a view from a slightly different point on the lens, so choosing $(u,v)$ moves the "
        "virtual camera. Between grid positions we blend the four nearest views bilinearly, which is correct when "
        "points move less than about a pixel per step and produces doubled edges beyond that (angular aliasing). "
        "The corner views show the largest available parallax; the path animation sweeps the camera round the aperture.",
        "05_perspective_shift.md")
    c1, c2 = st.columns([1, 2])
    with c1:
        u = st.slider("u (fractional row)", 0.0, float(L.nu - 1), float(L.center[0]), 0.1, key=K("pe_u"))
        v = st.slider("v (fractional column)", 0.0, float(L.nv - 1), float(L.center[1]), 0.1, key=K("pe_v"))
    with c2:
        show(aperture.synthesize_view(L, u, v), f"synthesised view at (u, v) = ({u:.1f}, {v:.1f})")
    st.markdown("**Corner views (2 x 2)**")
    cs = [(0, 0), (0, L.nv - 1), (L.nu - 1, 0), (L.nu - 1, L.nv - 1)]
    show(mosaic([L.views[a, b] for a, b in cs], 2, gap=4),
         "top-left, top-right (row 1); bottom-left, bottom-right (row 2): corners " + ", ".join(map(str, cs)))
    st.markdown("**Perspective path**")
    a, b, c = st.columns(3)
    path = a.selectbox("path", ["circle", "row", "column", "diagonal"], key=K("pe_path"))
    n = b.slider("frames", 8, 36, 20, key=K("pe_n"))
    rmax = float(min(L.nu - 1 - L.center[0], L.center[0], L.nv - 1 - L.center[1], L.center[1]))
    r = c.slider("radius (angular steps)", 0.5, max(rmax, 0.6), max(rmax, 0.6), 0.1, key=K("pe_r"))
    if st.button("Build perspective GIF", key=K("pe_btn")):
        st.session_state["pe_gif"] = (LK, c_persp_gif(L, LK, path, n, float(r)))
    g = st.session_state.get("pe_gif")
    if g and g[0] == LK:
        st.image(g[1], caption="perspective path", width="stretch")


# ==========================================================================
# 5. Depth and EDoF
# ==========================================================================
def _dmap(ax, d, vmin, vmax, title, cmap="viridis"):
    im = ax.imshow(d, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    return im


def tab_depth():
    explainer(
        "Depth is encoded in how a point moves between views, i.e. in its disparity. "
        "The defocus cue refocuses at many alphas and picks, per pixel, the alpha where the image is sharpest. "
        "The correspondence cue shears the light field and picks the alpha where all views agree (lowest variance). "
        "The EPI cue measures the local line slope with a structure tensor, with no search at all. "
        "The combined estimator fuses the first two and smooths edge-aware. "
        "For extended depth of field, the best-focused pixel from every focal-stack slice is merged into one sharp image.",
        "07_depth_estimation.md")
    c = st.columns(5)
    est = c[0].selectbox("estimator", ["defocus", "correspondence", "EPI structure tensor", "combined"], 3,
                         key=K("dp_est"))
    lo = c[1].number_input("alpha min", value=float(round(A_LO, 2)), step=0.25, key=K("dp_lo"))
    hi = c[2].number_input("alpha max", value=float(round(A_HI, 2)), step=0.25, key=K("dp_hi"))
    n = c[3].slider("alpha count", 5, 31, 11, key=K("dp_n"))
    win = c[4].slider("window", 3, 15, 7, 2, key=K("dp_w"))
    p = gate("dp_run", "Estimate depth", dict(est=est, lo=lo, hi=hi, n=n, win=win), LK)
    if p and p["hi"] > p["lo"]:
        r = c_depth(L, LK, p["est"], p["lo"], p["hi"], p["n"], p["win"])
        d = r["disparity"]
        vmin, vmax = (float(GT.min()), float(GT.max())) if GT is not None else (p["lo"], p["hi"])
        ncol = 4 if GT is not None else 3
        fig, axs = plt.subplots(1, ncol, figsize=(4 * ncol, 3.6))
        axs[0].imshow(np.clip(L.center_view, 0, 1).squeeze()); axs[0].set_title("centre view"); axs[0].axis("off")
        im = _dmap(axs[1], d, vmin, vmax, f"disparity ({p['est']})")
        fig.colorbar(im, ax=axs[1], fraction=0.046, label="disparity (px / view)")
        im = _dmap(axs[2], r["confidence"], 0, 1, "confidence", "gray")
        fig.colorbar(im, ax=axs[2], fraction=0.046)
        if GT is not None:
            im = _dmap(axs[3], GT, vmin, vmax, "ground truth")
            fig.colorbar(im, ax=axs[3], fraction=0.046)
        fig.tight_layout()
        show_fig(fig)
        if GT is not None:
            ev = depth.evaluate(d, GT)
            m = st.columns(4)
            m[0].metric("MAE", f"{ev['mae']:.3f}")
            m[1].metric("RMSE", f"{ev['rmse']:.3f}")
            m[2].metric("bad-pixel ratio (>0.25)", f"{100 * ev['bad_pixel_ratio']:.1f} %")
            m[3].metric("MAE incl. edges", f"{ev['mae_all']:.3f}")
            fig, ax = plt.subplots(figsize=(4.5, 3.6))
            im = _dmap(ax, np.abs(d - GT), 0, 1.0, "|error| (clipped at 1)", "magma")
            fig.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            show_fig(fig)
        else:
            st.caption("No ground-truth disparity for this dataset (only the synthetic scene has one).")

    st.subheader("All-in-focus (extended depth of field)")
    e = st.columns(4)
    sr = e[0].slider("stopped-down radius (views)", 0.0, float(max(L.nu, L.nv) // 2), 1.0, 0.5, key=K("ed_r"))
    ns = e[1].slider("noise sigma of full aperture", 0.0, 0.1, 0.02, 0.005, key=K("ed_noise"))
    fa = e[2].slider("focus alpha of the photos", float(round(A_LO, 2)), float(round(A_HI, 2)),
                     float(np.clip(0.0, A_LO, A_HI)), 0.05, key=K("ed_fa"))
    q = gate("ed_run", "Build all-in-focus + slide-58 comparison",
             dict(lo=lo, hi=hi, n=n, win=win, sr=sr, ns=ns, fa=fa), LK)
    if q and q["hi"] > q["lo"]:
        aif = c_aif(L, LK, q["lo"], q["hi"], q["n"], q["win"])
        full, stopped, aif2 = c_triptych(L, LK, q["lo"], q["hi"], q["n"], q["win"], q["sr"], q["ns"], q["fa"])
        cols = st.columns(3)
        with cols[0]:
            show(full, f"full aperture, focus alpha={q['fa']:.2f}: bright, shallow DoF")
        with cols[1]:
            show(stopped, f"stopped down r={q['sr']:g}: large DoF but dark/noisy")
        with cols[2]:
            show(aif2, "light field all-in-focus: sharp and clean")
        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        im = _dmap(ax, aif["disparity"], q["lo"], q["hi"], "selected focal-stack slice (alpha)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        show_fig(fig)


# ==========================================================================
# 6. Programmable aperture
# ==========================================================================
def tab_multiplex():
    explainer(
        "A programmable aperture opens a chosen set of angular cells per exposure, so N exposures give N views at full "
        "sensor resolution. Opening one cell at a time (identity) is dim and read-noise limited; opening about half the "
        "cells per shot (S-matrix rows) collects more light and the views are recovered by solving $y=Wx$. "
        "S-matrices only exist for certain sizes ($N=2^k-1$ or prime $\\equiv 3 \\bmod 4$), so for other sizes "
        "we search for a good 0/1 matrix by annealing. The demultiplexed views are compared with the clean ones by PSNR.",
        "08_programmable_aperture.md")
    opts = {}
    for n_ in (3, 5, 7):
        if n_ <= min(L.nu, L.nv):
            opts[f"{n_} x {n_}"] = (n_, n_)
    if L.nv >= 7:
        opts["1 x 7 row (S-matrix exists, N=7)"] = (1, 7)
    c = st.columns(4)
    names = list(opts)
    pick = c[0].selectbox("angular crop", names, min(1, len(names) - 1), key=K("mx_crop"))
    nu, nv = opts[pick]
    kind = c[1].selectbox("pattern type", ["S-matrix", "optimised", "random", "identity"], key=K("mx_kind"))
    iters = c[2].slider("optimise iterations", 200, 10000, 1500, 100, key=K("mx_it"))
    seed = c[3].number_input("seed", 0, 9999, 0, key=K("mx_seed"))
    c = st.columns(3)
    T = c[0].slider("total exposure T", 0.1, 4.0, 1.0, 0.1, key=K("mx_T"))
    photon = c[1].select_slider("photon scale (photons per unit)", [100, 300, 1000, 3000, 10000, 100000], 1000,
                                key=K("mx_ph"))
    read = c[2].slider("read noise sigma", 0.0, 0.1, 0.01, 0.002, key=K("mx_rd"))
    N = nu * nv
    if kind == "S-matrix" and N not in multiplex.valid_s_sizes(64):
        st.warning(f"No S-matrix of order {N} (valid orders: {multiplex.valid_s_sizes(40)}); using annealed patterns "
                   "instead. Pick the 1 x 7 row for a true S-matrix.")
    R = c_multiplex(L, LK, nu, nv, kind, iters, int(seed), T, photon, read)
    Lc = L.crop_angular(nu, nv)
    st.markdown("**Aperture patterns** (rows of W reshaped to the angular grid; white = open)")
    cols = st.columns(3)
    for col, (k, t) in zip(cols, (("identity", "identity"), ("mux", "multiplexed (selected)"), ("random", "random"))):
        with col:
            W = R[k]["W"]
            tiles = [np.pad(w.reshape(nu, nv), 0) for w in W]
            ncol = min(len(tiles), 7 if nu > 1 else 8)
            show(np.kron(mosaic(tiles, ncol, gap=1), np.ones((6, 6), np.float32)), f"{t} (K={W.shape[0]} shots)")
    st.markdown("**One captured shot per pattern type** (normalised by open area, shot 0)")
    cols = st.columns(3)
    for col, k in zip(cols, ("identity", "mux", "random")):
        with col:
            show(R[k]["shot"], k)
    st.markdown("**Demultiplexed centre view vs clean**")
    cols = st.columns(4)
    cu, cv = Lc.center
    with cols[0]:
        show(Lc.center_view, "clean view")
    for col, k in zip(cols[1:], ("identity", "mux", "random")):
        with col:
            show(R[k]["views"][cu, cv], f"{k}: PSNR {R[k]['psnr_c']:.1f} dB (all views {R[k]['psnr']:.1f})")
    st.markdown("**Closed-form theory** (slide 29: SNR relative to the plenoptic sample)")
    th = multiplex.theory_table(N, sigma_r=read, photon_scale=float(photon), total_exposure=T, W=R["mux"]["W"])
    rows = []
    for reg in ("photon", "read", "mixed"):
        for name in ("plenoptic", "pa_nomux", "pa_mux"):
            v = th[reg][name]
            rows.append({"regime": reg, "method": name, "mse": v["mse"], "snr_rel": v["snr_rel"],
                         "snr_rel_binned": v["snr_rel_binned"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Slide's claimed ratios (plenoptic, PA no-mux, PA mux): photon " +
               str(tuple(round(float(x), 3) for x in th["slide"]["photon"])) + ", read " +
               str(tuple(round(float(x), 3) for x in th["slide"]["read"])) + ".")
    st.markdown("**Refocus of the demultiplexed light field**")
    al = st.slider("alpha", float(round(A_LO, 2)), float(round(A_HI, 2)), float(np.clip(0.0, A_LO, A_HI)), 0.05,
                   key=K("mx_al"))
    X = lf.LightField(R["mux"]["views"])
    a, b = st.columns(2)
    with a:
        show(refocus.refocus(Lc, al), "clean light field")
    with b:
        show(refocus.refocus(X, al), "demultiplexed (selected pattern)")


# ==========================================================================
# 7. HDR
# ==========================================================================
def tab_hdr():
    explainer(
        "A sensor clips highlights and loses shadows in noise, and refocusing averages rays so one clipped highlight "
        "biases the whole photo. We give different sub-aperture views different exposures, warp them together using "
        "depth, and merge in the log-radiance domain with hat weights (Debevec-Malik). Alternatives: shoot the whole "
        "light field several times (C, the upper bound), merge in the refocused domain without depth, or vary the open "
        "aperture area (D). Always refocus on linear radiance and tonemap last.", "09_hdr_light_field.md")
    c = st.columns(4)
    source = c[0].selectbox("source", ["synthetic HDR scene", "pseudo-HDR from loaded dataset"], key=K("hd_src"))
    nv_h = c[1].select_slider("HDR synthetic views", [3, 5, 7], 5, key=K("hd_nv"))
    sz_h = c[2].select_slider("HDR synthetic size", [64, 96, 128], 96, key=K("hd_sz"))
    kind = c[3].selectbox("exposure pattern", ["checker", "rows", "radial", "random"], key=K("hd_kind"))
    R = c_hdr_source(source, LK, nv_h, sz_h, L)
    RK = f"{source}|{LK}|{nv_h}|{sz_h}"
    has_gt = R.meta.get("gt_disparity") is not None
    st.caption(f"radiance light field {R.nu}x{R.nv} views, {R.height}x{R.width}px, dynamic range "
               f"{R.meta.get('dynamic_range', float('nan')):.0f}:1" +
               (" (pseudo-HDR: not real HDR information)" if R.meta.get("pseudo_hdr") else ""))
    c = st.columns(4)
    e0 = c[0].slider("exposure 1 (stops)", -8, 8, -5, key=K("hd_e0"))
    e1 = c[1].slider("exposure 2 (stops)", -8, 8, 0, key=K("hd_e1"))
    e2 = c[2].slider("exposure 3 (stops)", -8, 8, 5, key=K("hd_e2"))
    exps = tuple(sorted(float(2.0 ** e) for e in (e0, e1, e2)))
    strategies = (["A (gt depth)"] if has_gt else []) + ["A (no depth)", "A (refocused domain)",
                                                          "C (temporal bracketing)", "D (aperture area)"]
    strat = c[3].selectbox("strategy", strategies, key=K("hd_strat"))
    c = st.columns(4)
    tm = c[0].selectbox("tonemap", ["reinhard", "log", "gamma"], key=K("hd_tm"))
    tkey = c[1].slider("tonemap key", 0.02, 0.8, 0.18, 0.01, key=K("hd_key"))
    tgam = c[2].slider("display gamma", 1.0, 3.0, 2.2, 0.1, key=K("hd_gam"))
    alpha = c[3].slider("refocus alpha", float(round(A_LO, 2)), float(round(A_HI, 2)),
                        float(np.clip(0.0, A_LO, A_HI)), 0.05, key=K("hd_al"))
    c = st.columns(2)
    photon = c[0].select_slider("capture photon scale", [200, 500, 2000, 10000], 2000, key=K("hd_ph"))
    cexp = c[1].slider("LDR comparison exposure (stops)", -8, 8, 0, key=K("hd_ce"))
    p = gate("hd_run", "Capture, merge and compare", dict(src=RK, strat=strat, kind=kind, exps=exps, tm=tm,
                                                          tkey=tkey, tgam=tgam, alpha=alpha, photon=photon, cexp=cexp), LK)
    if not p or p["src"] != RK:
        return
    res = c_hdr_run(R, RK, p["strat"], p["kind"], p["exps"], float(p["alpha"]), float(p["photon"]))
    gt_c = R.center_view
    lbar = float(np.exp(np.mean(np.log(1e-6 + gt_c.mean(-1)))))
    tmk = dict(method=p["tm"], key=p["tkey"], gamma=p["tgam"])
    T = lambda x: hdr.tonemap(x, lbar=lbar, **tmk)  # noqa: E731 - same exposure for gt and estimate
    st.markdown("**Bracketed LDR captures**")
    if "ldr" in res:
        ex = np.unique(res["em"])
        cu, cv = R.center
        cols = st.columns(len(ex) + 1)
        for col, t in zip(cols, ex):
            u_, v_ = np.argwhere(res["em"] == t)[0]
            with col:
                show(res["ldr"][u_, v_], f"view ({u_},{v_}), exposure {t:g}")
        with cols[-1]:
            show(np.kron(np.log2(res["em"]), np.ones((12, 12))) / (2 * max(abs(np.log2(res["em"])).max(), 1)) + 0.5,
                 "exposure map (log2)")
    elif "ldr_c" in res:
        cols = st.columns(len(res["exps"]))
        for col, im, t in zip(cols, res["ldr_c"], res["exps"]):
            with col:
                show(im, f"centre view, exposure {t:g}")
    else:
        cols = st.columns(len(res["exps"]))
        for col, im, t in zip(cols, res["ldr_d"], res["exps"]):
            with col:
                show(im, f"refocused photo, effective exposure {t:g}")
    st.markdown("**Merged HDR (tonemapped) vs ground truth**")
    if "lf" in res:
        est = res["lf"][R.center]
        gt_img = gt_c
        lab = "centre view"
    else:
        est = res["image"]
        gt_img = refocus.refocus(R, float(p["alpha"]))
        lab = f"refocused at alpha={p['alpha']:.2f}"
    a, b, m = st.columns([2, 2, 1])
    with a:
        show(T(est), f"merged HDR, {lab}")
    with b:
        show(T(gt_img), f"ground-truth radiance, {lab}")
    with m:
        st.metric("log-RMSE vs ground truth", f"{res['log_rmse']:.3f}",
                  help="natural-log RMSE of radiance; 0.69 = factor-2 error")
        st.metric("shots", res["shots"])
    st.markdown("**LDR vs HDR refocus**")
    est_views = res["lf"] if "lf" in res else None
    cmp_ = c_hdr_cmp(R, RK, float(p["alpha"]), float(2.0 ** p["cexp"]), est_views, float(p["photon"]))
    cols = st.columns(4)
    with cols[0]:
        show(cmp_["ldr"], f"LDR average (log-RMSE {cmp_['log_rmse_ldr']:.2f})")
    with cols[1]:
        show(cmp_["ldr_lin"], f"LDR, linearised first (log-RMSE {cmp_['log_rmse_ldr_lin']:.2f})")
    with cols[2]:
        show(cmp_["tm_first"], f"tonemap first, then refocus (PSNR {cmp_['psnr_display_tm_first']:.1f} dB)")
    with cols[3]:
        show(hdr.hdr_refocus(R, float(p["alpha"]), tonemap_kwargs=dict(lbar=lbar, **tmk)),
             f"HDR refocus (linear), tonemapped (log-RMSE {cmp_['log_rmse_hdr']:.2f})")
    st.caption("Difference between the naive LDR refocus and the HDR display at the same exposure:")
    show(np.clip(cmp_["diff"] * 4, 0, 1), "|LDR - HDR| x4")


# ==========================================================================
# assemble
# ==========================================================================
tabs = st.tabs(["Explore", "Plenoptic camera", "Refocus", "Perspective", "Depth and EDoF",
                "Programmable aperture", "HDR"])
for tab, (name, fn) in zip(tabs, [("Explore", tab_explore), ("Plenoptic camera", tab_plenoptic),
                                  ("Refocus", tab_refocus), ("Perspective", tab_perspective),
                                  ("Depth and EDoF", tab_depth), ("Programmable aperture", tab_multiplex),
                                  ("HDR", tab_hdr)]):
    with tab:
        safe(name, fn)
