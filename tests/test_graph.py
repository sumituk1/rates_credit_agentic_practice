from graph.workflow import build_graph


def test_graph_compiles():
    """Graph should compile without errors — no LLM or network calls."""
    app = build_graph()
    assert app is not None


def test_graph_has_expected_nodes():
    app = build_graph()
    nodes = set(app.nodes)
    assert {"hypothesis", "backtest", "evaluate", "critic"}.issubset(nodes)