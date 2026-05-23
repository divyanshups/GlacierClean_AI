"""
tests/conftest.py
Patches optional / network-unavailable packages with lightweight stubs
so that the unit tests can run in an offline / restricted environment.
"""
import sys
import types


def _stub_rich():
    rich = types.ModuleType("rich")
    console_mod = types.ModuleType("rich.console")
    logging_mod = types.ModuleType("rich.logging")
    panel_mod   = types.ModuleType("rich.panel")
    text_mod    = types.ModuleType("rich.text")
    table_mod   = types.ModuleType("rich.table")

    class Console:
        def print(self, *a, **kw): pass
        def rule(self, *a, **kw): pass

    class RichHandler:
        level = 0
        def __init__(self, **kw): pass
        def setLevel(self, lvl): self.level = lvl
        def emit(self, *a): pass
        def handle(self, *a): pass
        def createLock(self): pass

    class Panel:
        def __init__(self, *a, **kw): pass

    class Text:
        def __init__(self, *a, **kw): pass
        def append(self, *a, **kw): pass

    class Table:
        def __init__(self, **kw): pass
        def add_column(self, *a, **kw): pass
        def add_row(self, *a, **kw): pass

    console_mod.Console     = Console
    logging_mod.RichHandler = RichHandler
    panel_mod.Panel         = Panel
    text_mod.Text           = Text
    table_mod.Table         = Table

    rich.console = console_mod
    rich.logging = logging_mod
    rich.panel   = panel_mod
    rich.text    = text_mod
    rich.table   = table_mod

    for name, mod in [
        ("rich",         rich),
        ("rich.console", console_mod),
        ("rich.logging", logging_mod),
        ("rich.panel",   panel_mod),
        ("rich.text",    text_mod),
        ("rich.table",   table_mod),
    ]:
        sys.modules.setdefault(name, mod)


def _stub_plotly():
    class _Fig:
        def update_layout(self, **kw): return self
        def add_trace(self, *a, **kw): return self
        def show(self): pass

    go = types.ModuleType("plotly.graph_objects")
    go.Figure    = _Fig
    go.Heatmap   = lambda **kw: None
    go.Bar       = lambda **kw: None
    go.Box       = lambda **kw: None
    go.Histogram = lambda **kw: None
    go.Indicator = lambda **kw: None

    px = types.ModuleType("plotly.express")
    px.bar    = lambda **kw: _Fig()
    _colors   = types.SimpleNamespace(qualitative=types.SimpleNamespace(Bold=["#f00"] * 12))
    px.colors = _colors

    subplots = types.ModuleType("plotly.subplots")
    subplots.make_subplots = lambda **kw: _Fig()

    colors = types.ModuleType("plotly.colors")
    colors.qualitative = _colors.qualitative

    plotly = types.ModuleType("plotly")
    plotly.graph_objects = go
    plotly.express       = px
    plotly.subplots      = subplots
    plotly.colors        = colors

    for name, mod in [
        ("plotly",                plotly),
        ("plotly.graph_objects",  go),
        ("plotly.express",        px),
        ("plotly.subplots",       subplots),
        ("plotly.colors",         colors),
    ]:
        sys.modules.setdefault(name, mod)


_stub_rich()
_stub_plotly()
