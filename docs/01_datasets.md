# 01. Datasets

## 1. In one paragraph

The toolkit works on three kinds of 4-D light fields, all returned by `lf.io.load(name, **kw)` as an `lf.LightField` with `views[u, v, s, t, c]` (float32, [0, 1]): a fully synthetic scene with ground-truth disparity, a large-baseline gantry capture from the Stanford (new) Light Field Archive, and a real micro-lens capture from a Lytro Illum (Stanford Lytro archive). They cover the two ways of sampling the two-plane light field: a camera array / gantry (many big views, big baseline) and a plenoptic camera (few tiny views, tiny baseline).

## 2. Theory (what differs between the datasets)

| dataset | device | angular | spatial | baseline | file format |
|---|---|---|---|---|---|
| synthetic | none (numpy) | 9 x 9 (any) | 256 x 256 (any) | disparity -1 .. +1.5 px/step | generated |
| Lego Knights | gantry-mounted Canon camera, rectified | 17 x 17 | 1024 x 1024 | about 8.7 mm steps (about 140 mm total, from the file names) | 289 PNG, `out_RR_CC_<y_mm>_<x_mm>.png`, 8-bit RGB |
| Bracelet | same archive | 17 x 17 | 640 x 1024 (non-square) | gantry | same, but file names end in `_.png` (trailing underscore; the loader regex allows it) |
| Lytro flowers | Lytro Illum plenoptic camera | 14 x 14 (central 9 x 9 by default) | 376 x 541 micro-lenses | about the Illum aperture (millimetres) | one ESLF PNG, 14 x 14 tiles per micro-lens |

* Lego (and Bracelet): http://lightfield.stanford.edu/lfs.html. Cite the Stanford (New) Light Field Archive (Vaish, Adams, Levoy et al.; Stanford Computer Graphics Laboratory). See the site for terms of use; the data is intended for research.
* Lytro flowers: http://lightfields.stanford.edu/LF2016.html, file `flowers_plants_1_eslf.png` (category "flowers and plants"). Cite A. S. Raj, M. Lowney, R. Shah, G. Wetzstein, "Stanford Lytro Light Field Archive", 2016. The ESLF ("extracted sub-aperture light field") format is the decoded lenslet mosaic described in `docs/02_plenoptic_camera.md`.
* The Lytro Illum views are tiny-baseline: disparities are fractions of a pixel per step, so refocusing ranges are small; the gantry data has large steps and was downsampled for speed.
* Measured refocus ranges (`lf.refocus.SUGGESTED_ALPHA_RANGE`, from `sweep_sharpness`; disparity scales as 1/downsample):
  * Lego: (-2.0, 1.5) at downsample 4, 9x9 views. Castle wall sharp at -1, knights at 0, near floor at +1.
  * Bracelet: (-1.0, 2.0) at downsample 4, 9x9 views. Far clasp at 0, near chain at +1.0 to +1.5.
  * Lytro: (-1.5, 1.0) at full resolution, 9x9 views. Blossoms sharpest between -0.5 and +0.3.

## 3. Algorithm: loading and the download script

`python scripts/download_data.py [keywords]` downloads each dataset into its *own* directory `data/raw/<name>/` with `curl -C -` (resume, 5 retries), unzips archives after checking member names for absolute paths or `..`, deletes the zip and then writes an empty `.done` marker. A dataset with a `.done` marker is skipped, and `lf.io.available()` reports presence by that marker. Keywords select a subset (`lego`, `lytro`, `bracelet`).

## 4. Implementation notes

* `io.load('synthetic', n_views=9, size=256)`: calls `synthetic_light_field`.
* `io.load('lego', downsample=4, angular_crop=9)`: parses file names, orders axes so that near objects have positive disparity (camera x and y decrease with the angular index; see the Lego orientation note below), averages `downsample x downsample` blocks while reading (so memory never peaks at 3.6 GB), crops to the central `n x n` views.
* `io.load('lytro', crop_angular=9, downsample=1)`: de-interleaves the PNG with `reshape(H, 14, W, 14, C).transpose(1, 3, 0, 2, 4)`, then applies an automatic gain (`gain='auto'`) that maps the 99.5th percentile of the centre view to 1. The gain is stored in `meta['gain']` (about 2.3 for `flowers_plants_1`). The raw values are dark (mean about 0.1); `gain=None` keeps them unscaled.
* Disparity scales as `1/downsample`.
* Images are not gamma-linearised.

Memory as float32 (`views` array only):

| dataset | downsample 1 | 2 | 4 |
|---|---|---|---|
| synthetic 9x9, 256x256 | 64 MB | 16 MB | 4 MB |
| Lego 17x17, 1024x1024 | 3.64 GB | 0.91 GB | 0.23 GB |
| Lego cropped 9x9 | 1.02 GB | 0.25 GB | 64 MB |
| Lytro 14x14 376x541 (full) | 478 MB | 120 MB | 30 MB |
| Lytro cropped 9x9 | 198 MB | 49 MB | 12 MB |

## 5. Experiments / sanity checks

* Synthetic scene: three fronto-parallel layers, grass background (disparity -1.0), a text/brick square (0.0, in focus at the rectification plane) and a gravel disc (+1.5), front-most last. `lf.meta['gt_disparity']` holds the exact per-pixel disparity, which the depth module uses for error metrics. A refocus sweep is sharpest at -1 / 0 / +1.5 in the respective layers.
* Lego orientation check and Lytro de-interleave check: see the final section of `docs/02_plenoptic_camera.md` and the figures `lego_views_grid.png`, `lytro_views_grid.png`, `lytro_epi.png`. 
  **Lego orientation.** The loader orders the views so that the camera x and y coordinates both *decrease* with the angular index; the axis direction is fixed by our convention "nearer = positive disparity", not by the physical gantry motion (the file names give increasing/decreasing coordinates in the opposite sense). Verified empirically at `downsample=4, angular_crop=9`: shift-and-add with horizontal-only (`u = u0` row) and vertical-only (`v = v0` column) views peaks at the same alpha for the same region. Resulting alphas: the knights (rectification plane) at 0, the near floor at +1.0, the castle wall at -1.0.

## 6. Try it

```python
from lf import io
print(io.available())
lf = io.load('lego', downsample=4, angular_crop=9)   # about 4 s
lf = io.load('lytro')
```

## 7. Limitations and other public datasets

* Lego: static scene, single gantry; the ordering/sign conventions were fixed empirically (see above). Lytro: heavy vignetting at the outer views, sensor noise, one capture only.
* **HCI 4D Light Field Benchmark** (Honauer, Johannsen, Kondermann, Goldluecke, ACCV 2016): synthetic 9 x 9 scenes with ground-truth disparity; requires registration at http://hci-lightfield.iwr.uni-heidelberg.de. Natural next step for quantitative depth experiments.
* **EPFL Light-Field dataset / JPEG Pleno** (Rerabek and Ebrahimi, 2016): 118 Lytro Illum captures (raw `.LFR` and decoded) for compression and quality research.
* **Stanford Lytro archive** categories: flowers and plants, people, cars, buildings, fruits and vegetables, general scenes, occlusions, reflective surfaces, etc.; each as a raw `.LFR` and an ESLF PNG.
* Decoding raw `.LFR` files needs a calibrated decoder (Dansereau et al. 2013, "LF Toolbox").

See [dataset_links.md](dataset_links.md) for a verified list of download links to more scenes.

## 8. References

* A. S. Raj, M. Lowney, R. Shah, G. Wetzstein, "Stanford Lytro Light Field Archive", 2016.
* Stanford (New) Light Field Archive, http://lightfield.stanford.edu.
* K. Honauer, O. Johannsen, D. Kondermann, B. Goldluecke, "A dataset and evaluation methodology for depth estimation on 4D light fields", ACCV 2016.
* M. Rerabek, T. Ebrahimi, "New light field image dataset", QoMEX 2016.
* D. G. Dansereau, O. Pizarro, S. B. Williams, "Decoding, calibration and rectification for lenselet-based plenoptic cameras", CVPR 2013.
