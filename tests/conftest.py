# tests/conftest.py
import sys, pathlib

# Raiz do projeto = pasta que contém a pasta "core/"
ROOT = pathlib.Path(__file__).resolve().parents[1]

# Garante que a raiz esteja no PYTHONPATH durante os testes
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
