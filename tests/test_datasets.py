"""Catalogue + downloader, exercised against a local HTTP server (no internet)."""
import functools, http.server, io, threading, zipfile
import numpy as np, imageio.v3 as iio, pytest
import lf
from lf import datasets as ds


def test_catalogue_is_consistent():
    assert len(ds.STANFORD) == 12 and len(ds.LYTRO) == 9
    assert all(n in ds.DATASETS for n in ds.DEFAULT)
    assert all(s["url"].startswith("https://") and s["mb"] > 0 for s in ds.DATASETS.values())


def test_fetch_unzip_resume_and_load(tmp_path, monkeypatch):
    L = lf.synthetic.synthetic_light_field(n_views=3, size=32)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for u in range(3):
            for v in range(3):
                z.writestr(f"rectified/out_{u:02d}_{v:02d}_{-u:.6f}_{-v:.6f}_.png",
                           iio.imwrite("<bytes>", lf.utils.to_uint8(L.view(u, v)), extension=".png"))
    srv_dir = tmp_path / "srv"; srv_dir.mkdir(); (srv_dir / "rectified.zip").write_bytes(buf.getvalue())
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(srv_dir))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    raw = tmp_path / "raw"
    monkeypatch.setattr(ds, "RAW", raw); monkeypatch.setattr(lf.io, "RAW", raw)
    monkeypatch.setitem(ds.DATASETS, "stanford_tiny", {
        "url": f"http://127.0.0.1:{httpd.server_address[1]}/rectified.zip", "file": "rectified.zip",
        "unzip": True, "title": "tiny", "mb": 1, "key": "tiny"})
    calls = []
    try:
        d = ds.fetch("stanford_tiny", progress=lambda a, b: calls.append((a, b)))
    finally:
        httpd.shutdown()
    assert ds.is_downloaded("stanford_tiny") and calls and calls[-1][0] == calls[-1][1]
    assert not (d / "rectified.zip").exists() and len(list(d.rglob("out_*.png"))) == 9
    assert lf.io.available().get("tiny") is True
    out = lf.io.load("tiny")
    assert out.shape == L.shape and np.abs(out.views - L.views).max() < 1 / 255 + 1e-6
    assert ds.fetch("stanford_tiny") == d            # second call is a no-op
