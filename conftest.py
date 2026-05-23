"""Project-level test stubs for optional visualization/logging dependencies."""

import logging
import sys
import types


def _stub_rich() -> None:
    rich = types.ModuleType("rich")
    console_mod = types.ModuleType("rich.console")
    logging_mod = types.ModuleType("rich.logging")
    panel_mod = types.ModuleType("rich.panel")
    text_mod = types.ModuleType("rich.text")
    table_mod = types.ModuleType("rich.table")

    class Console:
        def print(self, *args, **kwargs):
            return None

        def rule(self, *args, **kwargs):
            return None

    class RichHandler(logging.Handler):
        def __init__(self, **kwargs):
            super().__init__()

        def emit(self, record):
            return None

    class Panel:
        def __init__(self, *args, **kwargs):
            return None

    class Text:
        def __init__(self, *args, **kwargs):
            self.parts = []

        def append(self, *args, **kwargs):
            return None

    class Table:
        def __init__(self, **kwargs):
            return None

        def add_column(self, *args, **kwargs):
            return None

        def add_row(self, *args, **kwargs):
            return None

    console_mod.Console = Console
    logging_mod.RichHandler = RichHandler
    panel_mod.Panel = Panel
    text_mod.Text = Text
    table_mod.Table = Table

    rich.console = console_mod
    rich.logging = logging_mod
    rich.panel = panel_mod
    rich.text = text_mod
    rich.table = table_mod

    for name, mod in [
        ("rich", rich),
        ("rich.console", console_mod),
        ("rich.logging", logging_mod),
        ("rich.panel", panel_mod),
        ("rich.text", text_mod),
        ("rich.table", table_mod),
    ]:
        sys.modules.setdefault(name, mod)


def _stub_plotly() -> None:
    class _Fig:
        def update_layout(self, **kwargs):
            return self

        def add_trace(self, *args, **kwargs):
            return self

        def show(self):
            return None

    go = types.ModuleType("plotly.graph_objects")
    go.Figure = _Fig
    go.Heatmap = lambda **kwargs: None
    go.Bar = lambda **kwargs: None
    go.Box = lambda **kwargs: None
    go.Histogram = lambda **kwargs: None
    go.Indicator = lambda **kwargs: None

    px = types.ModuleType("plotly.express")
    px.bar = lambda **kwargs: _Fig()
    colors_namespace = types.SimpleNamespace(
        qualitative=types.SimpleNamespace(Bold=["#f00"] * 12)
    )
    px.colors = colors_namespace

    subplots = types.ModuleType("plotly.subplots")
    subplots.make_subplots = lambda **kwargs: _Fig()

    colors = types.ModuleType("plotly.colors")
    colors.qualitative = colors_namespace.qualitative

    plotly = types.ModuleType("plotly")
    plotly.graph_objects = go
    plotly.express = px
    plotly.subplots = subplots
    plotly.colors = colors

    for name, mod in [
        ("plotly", plotly),
        ("plotly.graph_objects", go),
        ("plotly.express", px),
        ("plotly.subplots", subplots),
        ("plotly.colors", colors),
    ]:
        sys.modules.setdefault(name, mod)


_stub_rich()
_stub_plotly()
