# tests/test_accounts.py
import pandas as pd
from core.models import PNL_PREF_ORDER, load_current_long
def test_unknown_accounts():
    df = load_current_long()
    unk = set(df["indicador_id"].astype(str)) - set(PNL_PREF_ORDER)
    # tolere contas “novas” somente se houver regra clara de mapeamento
    assert not unk, f"Contas não mapeadas: {sorted(unk)}"
