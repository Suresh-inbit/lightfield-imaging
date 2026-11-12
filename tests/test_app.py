"""Smoke tests for the Streamlit explorer (streamlit.testing.v1.AppTest)."""
import pathlib

import pytest
from streamlit.testing.v1 import AppTest

import lf

APP = str(pathlib.Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def _app():
    return AppTest.from_file(APP, default_timeout=180)


def _slider(at, prefix):
    return [s for s in at.slider if s.label.startswith(prefix)][0]


def _click(at, label_part):
    b = [b for b in at.button if label_part in b.label][0]
    b.click()
    return at.run()


def test_default_synthetic_loads():
    at = _app().run()
    assert not at.exception
    assert at.selectbox(key="dataset").value == "synthetic"
    assert len(at.tabs) == 7


def test_refocus_alpha_slider_synthetic():
    at = _app().run()
    s = _slider(at, "alpha (refocus")
    at = s.set_value(1.5).run()
    assert not at.exception
    at = _slider(at, "alpha (refocus").set_value(-1.0).run()
    assert not at.exception


def test_heavy_buttons_synthetic():
    at = _app().run()
    for label in ("Estimate depth", "Build all-in-focus", "Show focal-stack strip",
                  "Build perspective GIF", "Animate sweep", "Capture, merge and compare"):
        at = _click(at, label)
        assert not at.exception, label
    assert not [e for e in at.error], [e.value for e in at.error]


def test_multiplex_options_synthetic():
    at = _app().run()
    sel = [s for s in at.selectbox if s.label == "pattern type"][0]
    for kind in ("optimised", "random", "identity", "S-matrix"):
        at = [s for s in at.selectbox if s.label == "pattern type"][0].select(kind).run()
        assert not at.exception, kind


@pytest.mark.skipif(not lf.io.available().get("lego"), reason="lego not downloaded")
def test_switch_to_lego():
    at = _app().run()
    at = at.selectbox(key="dataset").select("lego").run()
    assert not at.exception
    at = _slider(at, "alpha (refocus").set_value(-1.0).run()
    assert not at.exception


def test_upload_option_without_file():
    """Choosing 'upload' with no file shows instructions instead of failing."""
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=120).run()
    at.sidebar.selectbox(key="dataset").set_value("upload").run()
    assert not at.exception
    assert any("Accepted files" in i.value for i in at.info)


def test_upload_end_to_end(tmp_path):
    """Run the whole app on an 'uploaded' ZIP of views.  AppTest cannot drive
    st.file_uploader, so the widget is replaced by a stub returning the file."""
    import io, zipfile
    import imageio.v3 as iio
    L = lf.synthetic.synthetic_light_field(n_views=5, size=64)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for u in range(5):
            for v in range(5):
                z.writestr(f"v_{u:02d}_{v:02d}.png",
                           iio.imwrite("<bytes>", lf.utils.to_uint8(L.view(u, v)), extension=".png"))
    zpath = tmp_path / "views.zip"; zpath.write_bytes(buf.getvalue())
    app = pathlib.Path("app/streamlit_app.py").resolve()
    src = app.read_text().replace("from __future__ import annotations\n", "", 1)
    shim = f'''from __future__ import annotations
import sys, pathlib as _p, streamlit as _st
class _Up:
    name = "views.zip"
    def getvalue(self): return _p.Path({str(zpath)!r}).read_bytes()
_st.sidebar.file_uploader = lambda *a, **k: _Up()
_st.session_state.setdefault("dataset", "upload")
_st.session_state.setdefault("ds_up", 1)
__file__ = {str(app)!r}
sys.path.insert(0, {str(app.parent)!r})
'''
    script = tmp_path / "shim_app.py"; script.write_text(shim + src)
    at = AppTest.from_file(str(script), default_timeout=180).run()
    assert not at.exception and not at.error
    assert len(at.tabs) == 7
    assert any("5 x 5" in m.value and "64 x 64" in m.value for m in at.sidebar.markdown)


def test_download_option_lists_catalogue():
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=120).run()
    at.sidebar.selectbox(key="dataset").set_value("download").run()
    assert not at.exception
    assert len(at.selectbox(key="dl_pick").options) >= 1


def test_bundled_real_light_field_is_available_and_runs():
    """The repo ships a real Lytro light field, so the app never starts synthetic-only."""
    assert lf.io.available().get("lytro_people") is True
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=180).run()
    opts = at.sidebar.selectbox(key="dataset").options
    assert opts[0] == "synthetic" and opts[1] == "lytro_people"
    at.sidebar.selectbox(key="dataset").set_value("lytro_people").run()
    assert not at.exception and not at.error
    assert any("9 x 9" in m.value for m in at.sidebar.markdown)
