# tests/test_schema.py
from core.models import MONTHS_PT
def test_contract_columns(order_cols=["Conta", *MONTHS_PT, "Total Ano"]):
    assert order_cols[0] == "Conta" and order_cols[-1] == "Total Ano"
