#!/usr/bin/env python3
"""Download light-field datasets into ``data/raw/<name>/`` (one directory each).

    python scripts/download_data.py                # the three default sets (~780 MB)
    python scripts/download_data.py chess truck    # any sets, by keyword
    python scripts/download_data.py --list         # all keys, URLs and what is on disk

The catalogue lives in ``lf/datasets.py``; docs/dataset_links.md lists the links.
The app can fetch the same sets itself (dataset selector: "download from archive ...").
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lf.datasets import DATASETS, DEFAULT, fetch, is_downloaded   # noqa: E402


def _bar(done, total):
    pct = 100 * done / total if total else 0
    print(f"\r  {done / 1e6:7.0f} / {total / 1e6:.0f} MB  {pct:5.1f}%", end="", flush=True)


if __name__ == "__main__":
    keys = sys.argv[1:]
    if "--list" in keys:
        for name, spec in DATASETS.items():
            state = "on disk" if is_downloaded(name) else ("default" if name in DEFAULT else "optional")
            print(f"{name:28s} {state:9s} {spec['mb']:4d} MB  {spec['url']}")
        sys.exit(0)
    for name, spec in DATASETS.items():
        if (not keys and name not in DEFAULT) or (keys and not any(k in name for k in keys)):
            continue
        if is_downloaded(name):
            print(f"[skip] {name} already downloaded"); continue
        print(f"[get ] {name}  <-  {spec['url']}")
        fetch(name, progress=_bar)
        print(f"\n[ok  ] {name}")
