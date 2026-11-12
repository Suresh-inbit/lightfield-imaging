# Dataset download links

Every link below was checked on 2026-11-12 (HTTP 200; sizes from the server). The files work directly with this project: either through `scripts/download_data.py`, or by uploading them in the app (dataset selector, *upload your own ...*).

## 1. Stanford (New) Light Field Archive - gantry, 17 x 17 views

Index page: <https://lightfield.stanford.edu/lfs.html>. Each `rectified.zip` holds 289 PNGs named `out_RR_CC_y_x.png`. Upload the ZIP as-is (use spatial downsample 4 and a 9 x 9 crop), or fetch by key.

| Key | Scene | Size | Rectified ZIP |
|---|---|---|---|
| `lego` | Lego Knights | 373 MB | <https://lightfield.stanford.edu/data/lego_lf/rectified.zip> |
| `bracelet` | Bracelet | 229 MB | <https://lightfield.stanford.edu/data/bracelet_lf/rectified.zip> |
| `amethyst` | Amethyst | 283 MB | <https://lightfield.stanford.edu/data/amethyst_lf/rectified.zip> |
| `bulldozer` | Lego Bulldozer | 671 MB | <https://lightfield.stanford.edu/data/bulldozer_lf/rectified.zip> |
| `bunny` | Stanford Bunny | 363 MB | <https://lightfield.stanford.edu/data/bunny_lf/rectified.zip> |
| `chess` | Chess | 376 MB | <https://lightfield.stanford.edu/data/chess_lf/rectified.zip> |
| `gum_nuts` | Eucalyptus Flowers (gum nuts) | 680 MB | <https://lightfield.stanford.edu/data/gum_nuts_lf/rectified.zip> |
| `jelly_beans` | Jelly Beans | 126 MB | <https://lightfield.stanford.edu/data/jelly_beans_lf/rectified.zip> |
| `self_portrait` | Lego Gantry Self Portrait | 261 MB | <https://lightfield.stanford.edu/data/self_portrait_lf/rectified.zip> |
| `tarot_coarse` | Tarot Cards and Crystal Ball (large angular extent) | 496 MB | <https://lightfield.stanford.edu/data/tarot_coarse_lf/rectified.zip> |
| `tarot_fine` | Tarot Cards and Crystal Ball (small angular extent) | 475 MB | <https://lightfield.stanford.edu/data/tarot_fine_lf/rectified.zip> |
| `truck` | Lego Truck | 444 MB | <https://lightfield.stanford.edu/data/truck_lf/rectified.zip> |

```bash
python scripts/download_data.py chess truck      # any keys from the table
python scripts/download_data.py --list           # show all keys and what is on disk
```
```python
import lf
L = lf.io.load('chess', downsample=4, angular_crop=9)   # any downloaded key
```

## 2. Stanford Lytro Light Field Archive - Lytro Illum, 14 x 14 lenslet (ESLF) PNGs

Index page: <https://lightfields.stanford.edu/LF2016.html>. Single images follow the pattern `https://lightfields.stanford.edu/images/<category>/raw/<category>_<n>_eslf.png`; each category page lists its images (for example flowers_plants has 65). Upload a PNG in the app as a *lenslet image* with 14 x 14 pixels per lens.

| Category | Sample image (`_1_`) | Size | Whole category ZIP | ZIP size |
|---|---|---|---|---|
| [bikes](https://lightfields.stanford.edu/bikes.html) | <https://lightfields.stanford.edu/images/bikes/raw/bikes_1_eslf.png> | 166 MB | <https://lightfields.stanford.edu/images/bikes.zip> | 3.7 GB |
| [buildings](https://lightfields.stanford.edu/buildings.html) | <https://lightfields.stanford.edu/images/buildings/raw/buildings_1_eslf.png> | 187 MB | <https://lightfields.stanford.edu/images/buildings.zip> | 6.4 GB |
| [cars](https://lightfields.stanford.edu/cars.html) | <https://lightfields.stanford.edu/images/cars/raw/cars_1_eslf.png> | 183 MB | <https://lightfields.stanford.edu/images/cars.zip> | 10.4 GB |
| [flowers_plants](https://lightfields.stanford.edu/flowers_plants.html) | <https://lightfields.stanford.edu/images/flowers_plants/raw/flowers_plants_1_eslf.png> | 187 MB | <https://lightfields.stanford.edu/images/flowers_plants.zip> | 12.1 GB |
| [fruits_vegetables](https://lightfields.stanford.edu/fruits_vegetables.html) | <https://lightfields.stanford.edu/images/fruits_vegetables/raw/fruits_vegetables_1_eslf.png> | 180 MB | <https://lightfields.stanford.edu/images/fruits_vegetables.zip> | 3.6 GB |
| [general](https://lightfields.stanford.edu/general.html) | <https://lightfields.stanford.edu/images/general/raw/general_1_eslf.png> | 211 MB | <https://lightfields.stanford.edu/images/general.zip> | 10.4 GB |
| [occlusions](https://lightfields.stanford.edu/occlusions.html) | <https://lightfields.stanford.edu/images/occlusions/raw/occlusions_1_eslf.png> | 214 MB | <https://lightfields.stanford.edu/images/occlusions.zip> | 9.7 GB |
| [people](https://lightfields.stanford.edu/people.html) | <https://lightfields.stanford.edu/images/people/raw/people_1_eslf.png> | 180 MB | <https://lightfields.stanford.edu/images/people.zip> | 3.1 GB |
| [reflective](https://lightfields.stanford.edu/reflective.html) | <https://lightfields.stanford.edu/images/reflective/raw/reflective_1_eslf.png> | 208 MB | <https://lightfields.stanford.edu/images/reflective.zip> | 5.9 GB |

The category ZIPs are large; prefer single ESLF PNGs. `python scripts/download_data.py lytro` fetches `flowers_plants_1`; `lytro_<category>` (for example `lytro_reflective`) fetches that category's sample image.

## 3. Other public light-field datasets (not wired into the loaders)

| Dataset | What it is | Access |
|---|---|---|
| [HCI 4D Light Field Benchmark](https://lightfield-analysis.uni-konstanz.de/) | synthetic 9 x 9 scenes with ground-truth depth (Honauer et al., ACCV 2016) | free, but the download link is sent after registering an e-mail address |
| [HCI tools and resources](https://github.com/lightfield-analysis/resources) | file readers and evaluation code for the benchmark | GitHub |
| [EPFL Light-Field Image Dataset](https://www.epfl.ch/labs/mmspg/downloads/epfl-light-field-image-dataset/) | 118 Lytro Illum captures (the JPEG Pleno set), raw `.LFR` files | download page; raw files need a Lytro decoder |
| [INRIA synthetic light fields](http://clim.inria.fr/Datasets/InriaSynLF/index.html) | dense and sparse synthetic light fields with depth | download page |

To use one of these, export the views as images named `<anything>_RR_CC.png` in a ZIP, or as a `(U, V, H, W, C)` `.npy` array, and upload it.

## 4. Citing the data

* Stanford (New) Light Field Archive: V. Vaish, A. Adams, Stanford Computer Graphics Laboratory, 2008.
* A. S. Raj, M. Lowney, R. Shah, G. Wetzstein, "Stanford Lytro Light Field Archive", 2016.
* K. Honauer, O. Johannsen, D. Kondermann, B. Goldluecke, "A dataset and evaluation methodology for depth estimation on 4D light fields", ACCV 2016.
* M. Rerabek, T. Ebrahimi, "New light field image dataset", QoMEX 2016.
