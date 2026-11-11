"""Figures and results table for docs/09_hdr_light_field.md (headless)."""
import sys, time, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lf import hdr as H, refocus as R, io as lfio, LightField

OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
EXPS = (1 / 32, 1.0, 32.0)
N, SIZE = 7, 128
T0 = time.time()


def tm(x):  # display tonemap for radiance
    return H.tonemap(x, "reinhard")


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, dpi=110); plt.close(fig)
    print("wrote", name, f"({time.time() - T0:.0f}s)")


def show(ax, img, title="", **kw):
    ax.imshow(img, **kw); ax.set_title(title, fontsize=9); ax.axis("off")


gt = H.synthetic_hdr_light_field(N, SIZE, seed=0)
d_gt = gt.meta["gt_disparity"]
c = gt.center
print("dynamic range", gt.meta["dynamic_range"])
rng = np.random.default_rng(0)

# ---- 1. scene -------------------------------------------------------------
fig, ax = plt.subplots(1, 4, figsize=(14, 3.8))
caps = [H.simulate_capture(gt.center_view, t, rng=rng) for t in EXPS]
for a, t, im in zip(ax, EXPS, caps):
    show(a, im, f"exposure x{t:g} (LDR)")
show(ax[3], tm(gt.center_view), f"tonemapped radiance ({gt.meta['dynamic_range']:.0f}:1)")
save(fig, "hdr_synthetic_scene.png")

# ---- 2. response curve ----------------------------------------------------
ex5 = [1 / 32, 1 / 8, 1 / 2, 2, 8]
imgs = [H.simulate_capture(gt.center_view, t, rng=rng) for t in ex5]
g, _ = H.solve_response(H.sample_pixels(imgs, 200, rng), ex5)
z = np.arange(256); true = 2.2 * np.log(np.maximum(z, 1) / 255.0)
sel = (z > 8) & (z < 250)
off = np.mean(g[sel] - true[sel])
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(z, true + off, "k--", label="true  2.2 ln(z/255) + offset")
ax.plot(z, g, "r", label="recovered g(Z)")
ax.set_xlabel("pixel value Z"); ax.set_ylabel("g = ln E + ln dt"); ax.legend(); ax.set_title(
    f"Debevec-Malik response (corr {np.corrcoef(g[sel], true[sel])[0, 1]:.4f})", fontsize=9)
save(fig, "hdr_response_curve.png")
resp_corr = np.corrcoef(g[sel], true[sel])[0, 1]

# ---- 3. patterns ----------------------------------------------------------
fig, ax = plt.subplots(1, 4, figsize=(11, 3))
for a, k in zip(ax, ["checker", "rows", "radial", "random"]):
    m = H.exposure_pattern(N, N, k, EXPS)
    im = a.imshow(np.log2(m), cmap="coolwarm", vmin=-5, vmax=5); a.set_title(k); a.set_xlabel("v"); a.set_ylabel("u")
fig.colorbar(im, ax=ax, label="log2 exposure", shrink=0.8)
fig.savefig(OUT / "hdr_exposure_patterns.png", dpi=110); plt.close(fig); print("wrote hdr_exposure_patterns.png")

# ---- experiments ----------------------------------------------------------
def refocused_err(hdr_lf, alpha):
    return H.log_rmse(R.refocus(hdr_lf, alpha), R.refocus(gt, alpha))

rows = []   # name, shots, lf log-rmse, refocus a=0, refocus a=1.5
res = {}
for kind in ["checker", "rows", "radial", "random"]:
    r = H.strategy_a(gt, kind, rng=np.random.default_rng(1)); res[kind] = r
    rows.append((f"A {kind}, gt depth", 1, r["log_rmse"], refocused_err(r["hdr_lf"], 0), refocused_err(r["hdr_lf"], 1.5)))
rn = H.strategy_a(gt, "checker", disparity=None, rng=np.random.default_rng(1)); res["nodepth"] = rn
rows.append(("A checker, no depth", 1, rn["log_rmse"], refocused_err(rn["hdr_lf"], 0), refocused_err(rn["hdr_lf"], 1.5)))
# A merged in the refocused domain (depth-free)
ra = res["checker"]
ref_dom = {a: H.log_rmse(H.merge_bracketed_refocused(ra["ldr_lf"], ra["exposure_map"], a), R.refocus(gt, a)) for a in (0.0, 1.5)}
rows.append(("A checker, refocused-domain merge", 1, np.nan, ref_dom[0.0], ref_dom[1.5]))
# estimated depth (optional)
est_row = None
try:
    from lf import depth as D
    ldr = ra["ldr_lf"]; em = ra["exposure_map"]
    E = np.log(np.maximum(H.inverse_crf("gamma")(ldr.views) / em[..., None, None, None], 1e-3))
    norm = LightField(((E - np.log(1e-3)) / np.log(1e5)).clip(0, 1).astype(np.float32))
    dest = D.depth_combined(norm, np.linspace(-1.5, 2.0, 15))["disparity"]
    ev = D.evaluate(dest, d_gt)
    re = H.strategy_a(gt, "checker", disparity=dest, rng=np.random.default_rng(1)); res["est"] = re
    est_row = (f"A checker, estimated depth (MAE {ev['mae']:.2f}px)", 1, re["log_rmse"],
               refocused_err(re["hdr_lf"], 0), refocused_err(re["hdr_lf"], 1.5))
    rows.append(est_row)
except Exception as e:  # never depend on lf.depth
    print("estimated-depth run skipped:", repr(e))
cc = H.temporal_bracket_hdr_lf(gt, EXPS, rng=np.random.default_rng(1))
rows.append(("C temporal bracket (3 shots)", 3, cc["log_rmse"], refocused_err(cc["hdr_lf"], 0), refocused_err(cc["hdr_lf"], 1.5)))
for t in EXPS:
    l, _ = H.capture_bracketed_views(gt, np.full((N, N), t), rng=np.random.default_rng(1))
    s = LightField(H.inverse_crf("gamma")(l.views) / t)
    rows.append((f"single exposure x{t:g}", 1, H.evaluate_hdr_lf(s, gt)["mean"], refocused_err(s, 0), refocused_err(s, 1.5)))
dd = {a: H.aperture_area_bracketing(gt, (1, 3, 9), alpha=a, rng=np.random.default_rng(1)) for a in (0.0, 1.5)}
rows.append(("D aperture area 1/3/9 cells (3 shots)", 3, np.nan, dd[0.0]["log_rmse"], dd[1.5]["log_rmse"]))
# B
b0 = H.ldr_refocus_comparison(gt, 0.0, 1.0, hdr_lf=cc["hdr_lf"], rng=np.random.default_rng(1))
b1 = H.ldr_refocus_comparison(gt, 1.5, 1.0, hdr_lf=cc["hdr_lf"], rng=np.random.default_rng(1))
rows.append(("B refocus LDR (exp 1, display avg)", 1, np.nan, b0["log_rmse_ldr"], b1["log_rmse_ldr"]))
rows.append(("B refocus LDR linearised avg", 1, np.nan, b0["log_rmse_ldr_lin"], b1["log_rmse_ldr_lin"]))
rows.append(("B refocus HDR (from C) in radiance", 3, np.nan, b0["log_rmse_hdr"], b1["log_rmse_hdr"]))

md = ["| method | shots | LF log-RMSE | refocus a=0 | refocus a=1.5 |", "|---|---|---|---|---|"]
f = lambda x: "-" if np.isnan(x) else f"{x:.3f}"
for r in rows:
    md.append(f"| {r[0]} | {r[1]} | {f(r[2])} | {f(r[3])} | {f(r[4])} |")
(OUT / "hdr_strategy_table.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
fig, ax = plt.subplots(figsize=(10, 0.38 * len(rows) + 1)); ax.axis("off")
tb = ax.table(cellText=[[r[0], r[1], f(r[2]), f(r[3]), f(r[4])] for r in rows],
              colLabels=["method", "shots", "LF log-RMSE", "refocus a=0", "refocus a=1.5"], loc="center", cellLoc="left")
tb.auto_set_font_size(False); tb.set_fontsize(8); tb.auto_set_column_width(range(5))
ax.set_title("Log-RMSE (natural log) against ground-truth radiance; lower is better", fontsize=9)
save(fig, "hdr_strategy_table.png")

# ---- 4. strategy A --------------------------------------------------------
ldr = res["checker"]["ldr_lf"]
mont = np.concatenate([np.concatenate([ldr.views[u, v, ::2, ::2] for v in range(N)], 1) for u in range(N)], 0)
fig, ax = plt.subplots(2, 3, figsize=(13, 8.5))
show(ax[0, 0], mont, "7x7 bracketed LDR views (checker exposures)")
hA, hN = res["checker"]["hdr_lf"], res["nodepth"]["hdr_lf"]
err = lambda h: np.abs(np.log(h.center_view + 1e-4) - np.log(gt.center_view + 1e-4)).mean(-1)
show(ax[0, 1], tm(hA.center_view), f"merged HDR centre view, gt depth\nlog-RMSE {H.log_rmse(hA.center_view, gt.center_view):.2f}")
show(ax[0, 2], err(hA), "|ln error|, gt depth", cmap="magma", vmin=0, vmax=3)
show(ax[1, 0], tm(gt.center_view), "ground-truth radiance (tonemapped)")
show(ax[1, 1], tm(hN.center_view), f"merged HDR centre view, NO depth\nlog-RMSE {H.log_rmse(hN.center_view, gt.center_view):.2f}")
show(ax[1, 2], err(hN), "|ln error|, no depth", cmap="magma", vmin=0, vmax=3)
save(fig, "hdr_strategyA.png")

# ---- 5. refocus LDR vs HDR ------------------------------------------------
alphas = [-1.0, 0.0, 1.5]
fig, ax = plt.subplots(3, 4, figsize=(14, 10.5))
for i, a in enumerate(alphas):
    r = H.ldr_refocus_comparison(gt, a, 1.0, hdr_lf=cc["hdr_lf"], rng=np.random.default_rng(1))
    show(ax[i, 0], r["ldr"], f"alpha={a}: capture-clip then refocus\nlog-RMSE {r['log_rmse_ldr']:.2f}")
    show(ax[i, 1], r["tm_first"], f"tonemap then refocus\nPSNR {r['psnr_display_tm_first']:.1f} dB")
    show(ax[i, 2], tm(r["hdr"]), f"refocus radiance (from C), then tonemap\nlog-RMSE {r['log_rmse_hdr']:.2f}")
    show(ax[i, 3], tm(r["gt"]), "ground truth")
save(fig, "hdr_refocus_ldr_vs_hdr.png")

# ---- 7. exposure fusion ---------------------------------------------------
cap3 = [H.simulate_capture(gt.center_view, t, rng=np.random.default_rng(2)) for t in EXPS]
fus = H.exposure_fusion(cap3)
mrg = H.merge_hdr(cap3, EXPS, response="gamma")
fig, ax = plt.subplots(1, 4, figsize=(14, 3.8))
for a, t, im in zip(ax[:0], EXPS, cap3):
    pass
show(ax[0], np.concatenate(cap3, 1), "three input exposures")
show(ax[1], fus, "Mertens exposure fusion\n(no response curve)")
show(ax[2], tm(mrg), f"Debevec-Malik merge + Reinhard\nlog-RMSE {H.log_rmse(mrg, gt.center_view):.2f}")
show(ax[3], tm(gt.center_view), "ground truth (tonemapped)")
save(fig, "hdr_exposure_fusion.png")

# ---- 8. real data ---------------------------------------------------------
if lfio.available().get("lego"):
    try:
        L = lfio.load("lego", downsample=4, angular_crop=7)
        P = H.hdr_from_ldr_light_field(L)
        fig, ax = plt.subplots(2, 3, figsize=(13, 8.5))
        a0 = 0.0
        ldr_r = R.refocus(L, a0)
        pr = R.refocus(P, a0)
        show(ax[0, 0], L.center_view, "LDR centre view")
        show(ax[0, 1], tm(P.center_view), "pseudo-radiance centre view (tonemapped)")
        show(ax[0, 2], ldr_r, "refocus in display domain (LDR)")
        show(ax[1, 0], np.clip(P.center_view, 0, 1) ** (1 / 2.2), "pseudo-HDR clipped at 1 (gamma)")
        show(ax[1, 1], tm(pr), "refocus in radiance, then tonemap")
        show(ax[1, 2], np.abs(tm(pr) - ldr_r).mean(-1), "difference", cmap="magma")
        fig.suptitle("Lego Knights, PSEUDO-HDR (inverse gamma + boosted top 2%); illustration only", fontsize=10)
        save(fig, "hdr_lego_pseudo.png")
    except Exception as e:
        print("lego figure skipped:", repr(e))
else:
    print("lego not available: synthetic figures only")
print(f"resp corr {resp_corr:.4f}; total {time.time() - T0:.0f}s")
