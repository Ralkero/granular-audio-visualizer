"""Fixtures shared by the tests that draw."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app):
    from gmv.visualizer.window import VisualizerWindow

    view = VisualizerWindow()
    view.timer.stop()  # the tests drive the refresh themselves
    view.resize(1280, 800)
    yield view
    view.close()
