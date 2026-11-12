"""Catalogue of downloadable light fields and a streaming downloader.

Used by ``scripts/download_data.py`` and by the app's "download from archive"
option.  Each dataset lands in its own directory ``data/raw/<name>/`` and is
complete when ``data/raw/<name>/.done`` exists.  See docs/dataset_links.md.
"""
from __future__ import annotations
import pathlib, urllib.request, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# key -> (folder on lightfield.stanford.edu, title, size in MB)
STANFORD = {
    "lego": ("lego_lf", "Lego Knights", 373),
    "bracelet": ("bracelet_lf", "Bracelet", 229),
    "amethyst": ("amethyst_lf", "Amethyst", 283),
    "bulldozer": ("bulldozer_lf", "Lego Bulldozer", 671),
    "bunny": ("bunny_lf", "Stanford Bunny", 363),
    "chess": ("chess_lf", "Chess", 376),
    "gum_nuts": ("gum_nuts_lf", "Eucalyptus Flowers", 680),
    "jelly_beans": ("jelly_beans_lf", "Jelly Beans", 126),
    "self_portrait": ("self_portrait_lf", "Lego Gantry Self Portrait", 261),
    "tarot_coarse": ("tarot_coarse_lf", "Tarot Cards (large angular extent)", 496),
    "tarot_fine": ("tarot_fine_lf", "Tarot Cards (small angular extent)", 475),
    "truck": ("truck_lf", "Lego Truck", 444),
}
# Lytro Illum categories -> size in MB of the sample image <category>_1_eslf.png
LYTRO = {"bikes": 166, "buildings": 187, "cars": 183, "flowers_plants": 187, "fruits_vegetables": 180,
         "general": 211, "occlusions": 214, "people": 180, "reflective": 208}

DATASETS: dict[str, dict] = {}
for _k, (_f, _t, _mb) in STANFORD.items():
    DATASETS["stanford_" + ("lego_knights" if _k == "lego" else _k)] = {
        "url": f"https://lightfield.stanford.edu/data/{_f}/rectified.zip", "file": "rectified.zip",
        "unzip": True, "title": f"Stanford gantry: {_t}", "mb": _mb, "key": _k}
for _c, _mb in LYTRO.items():
    _name = "lytro_illum_flowers" if _c == "flowers_plants" else "lytro_" + _c
    DATASETS[_name] = {"url": f"https://lightfields.stanford.edu/images/{_c}/raw/{_c}_1_eslf.png",
                       "file": f"{_c}_1_eslf.png", "unzip": False, "title": f"Lytro Illum: {_c} (sample 1)",
                       "mb": _mb, "key": "lytro" if _c == "flowers_plants" else "lytro_" + _c}

# fetched by the script when no keyword is given (~780 MB)
DEFAULT = ("stanford_lego_knights", "lytro_illum_flowers", "stanford_bracelet")


def is_downloaded(name: str) -> bool:
    return (RAW / name / ".done").exists()


def fetch(name: str, progress=None, chunk: int = 1 << 20) -> pathlib.Path:
    """Download (and unzip) one catalogue entry; returns its directory.

    ``progress(done_bytes, total_bytes)`` is called about once per MB.  An
    interrupted download resumes from the partial file (HTTP Range).  ZIP
    members are checked against path traversal before extraction."""
    spec = DATASETS[name]
    d = RAW / name
    d.mkdir(parents=True, exist_ok=True)
    if is_downloaded(name):
        return d
    out = d / spec["file"]
    part = d / (spec["file"] + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(spec["url"], headers={"User-Agent": "lightfield-toolkit",
                                                        **({"Range": f"bytes={have}-"} if have else {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:          # server ignored the range: start again
            have = 0
        total = have + int(r.headers.get("Content-Length") or 0)
        with open(part, "ab" if have else "wb") as f:
            done = have
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                f.write(buf); done += len(buf)
                if progress:
                    progress(done, total)
    if total and part.stat().st_size < total:
        raise IOError(f"download of {name} incomplete ({part.stat().st_size} of {total} bytes); run again to resume")
    part.rename(out)
    if spec["unzip"]:
        with zipfile.ZipFile(out) as z:
            for m in z.namelist():
                if m.startswith("/") or ".." in pathlib.PurePosixPath(m).parts:
                    raise RuntimeError(f"unsafe ZIP member {m}")
            z.extractall(d)
        out.unlink()
    (d / ".done").touch()
    return d
