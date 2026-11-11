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
    python scripts/download_data.py                # everything
    python scripts/download_data.py lego lytro     # subset by keyword
"""
import os, sys, subprocess, zipfile, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

DATASETS = {
    "stanford_lego_knights": {
        "url": "https://lightfield.stanford.edu/data/lego_lf/rectified.zip",
        "file": "rectified.zip", "unzip": True},
    "lytro_illum_flowers": {
        "url": "https://lightfields.stanford.edu/images/flowers_plants/raw/flowers_plants_1_eslf.png",
        "file": "flowers_plants_1_eslf.png", "unzip": False},
    "stanford_bracelet": {
        "url": "https://lightfield.stanford.edu/data/bracelet_lf/rectified.zip",
        "file": "rectified.zip", "unzip": True},
}

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
    for name, spec in DATASETS.items():
        if keys and not any(k in name for k in keys):
            continue
        fetch(name, spec)
