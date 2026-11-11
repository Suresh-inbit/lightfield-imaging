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
