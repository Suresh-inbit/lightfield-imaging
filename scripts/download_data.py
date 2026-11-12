#!/usr/bin/env python3
"""Download the light-field datasets used in this project.

Each dataset lands in its own directory under ``data/raw/<name>/`` so that
untrusted downloaded content never mixes with project code.

Datasets
--------
stanford_lego_knights : Stanford (New) Light Field Archive, "Lego Knights".
    17 x 17 views, 1024 x 1024 px, captured with a gantry-mounted camera.
    Rectified PNGs. http://lightfield.stanford.edu/lfs.html
stanford_bracelet : Same archive, "Bracelet" (17 x 17 views).
lytro_illum_flowers : Stanford Lytro Light Field Archive (Raj, Lowney,
    Shah, Wetzstein 2016). One ESLF ("extracted sub-aperture light field")
    PNG from a Lytro Illum plenoptic camera: 14 x 14 angular samples
    interleaved per micro-lens, 376 x 541 micro-lenses.
    http://lightfields.stanford.edu/LF2016.html

Usage
-----
    python scripts/download_data.py                # the three default sets
    python scripts/download_data.py chess truck    # any sets, by keyword
    python scripts/download_data.py --list         # all keys, URLs and what is on disk
See docs/dataset_links.md for the full list of links.
"""
import os, sys, subprocess, zipfile, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

STANFORD = {  # key -> folder on lightfield.stanford.edu
    "lego": "lego_lf",
    "bracelet": "bracelet_lf",
    "amethyst": "amethyst_lf",
    "bulldozer": "bulldozer_lf",
    "bunny": "bunny_lf",
    "chess": "chess_lf",
    "gum_nuts": "gum_nuts_lf",
    "jelly_beans": "jelly_beans_lf",
    "self_portrait": "self_portrait_lf",
    "tarot_coarse": "tarot_coarse_lf",
    "tarot_fine": "tarot_fine_lf",
    "truck": "truck_lf",
}
LYTRO_CATEGORIES = ['bikes', 'buildings', 'cars', 'flowers_plants', 'fruits_vegetables', 'general', 'occlusions', 'people', 'reflective']

DATASETS = {}
for _k, _f in STANFORD.items():
    DATASETS["stanford_" + ("lego_knights" if _k == "lego" else _k)] = {
        "url": f"https://lightfield.stanford.edu/data/{_f}/rectified.zip", "file": "rectified.zip", "unzip": True}
DATASETS["lytro_illum_flowers"] = {
    "url": "https://lightfields.stanford.edu/images/flowers_plants/raw/flowers_plants_1_eslf.png",
    "file": "flowers_plants_1_eslf.png", "unzip": False}
for _c in LYTRO_CATEGORIES:
    if _c != "flowers_plants":
        DATASETS["lytro_" + _c] = {"url": f"https://lightfields.stanford.edu/images/{_c}/raw/{_c}_1_eslf.png",
                                   "file": f"{_c}_1_eslf.png", "unzip": False}

# fetched when no keyword is given (~780 MB); everything else is opt-in by keyword
DEFAULT = ("stanford_lego_knights", "lytro_illum_flowers", "stanford_bracelet")

def fetch(name, spec):
    d = RAW / name
    d.mkdir(parents=True, exist_ok=True)
    out = d / spec["file"]
    done = d / ".done"
    if done.exists():
        print(f"[skip] {name} already downloaded"); return
    print(f"[get ] {name}  <-  {spec['url']}", flush=True)
    # curl with resume so an interrupted run can continue
    subprocess.run(["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "5",
                    "-C", "-", "-o", str(out), spec["url"]], check=True)
    if spec["unzip"]:
        print(f"[zip ] extracting {out.name}", flush=True)
        with zipfile.ZipFile(out) as z:
            for m in z.namelist():                 # guard against path traversal
                if m.startswith("/") or ".." in pathlib.Path(m).parts:
                    raise RuntimeError(f"unsafe member {m}")
            z.extractall(d)
        out.unlink()
    done.touch()
    print(f"[ok  ] {name}", flush=True)

if __name__ == "__main__":
    keys = sys.argv[1:]
    if "--list" in keys:
        for name, spec in DATASETS.items():
            state = "on disk" if (RAW / name / ".done").exists() else ("default" if name in DEFAULT else "optional")
            print(f"{name:28s} {state:9s} {spec['url']}")
        sys.exit(0)
    for name, spec in DATASETS.items():
        if (not keys and name not in DEFAULT) or (keys and not any(k in name for k in keys)):
            continue
        fetch(name, spec)
