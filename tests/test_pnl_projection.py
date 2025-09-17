"""Wrapper para executar o teste de projeção definido em teste_pnl.py dentro da suíte padrão."""
from teste_pnl import test_projection_volumes_override


def test_projection_volumes_override_wrapper(monkeypatch):
    test_projection_volumes_override(monkeypatch)
