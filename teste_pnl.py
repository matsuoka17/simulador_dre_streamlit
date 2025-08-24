# teste_pnl.py
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Teste P&L", layout="wide")
st.title("Validador do P&L • Smoke Test")

# Imports
try:
    import core.paths as P
    import core.models as M
    import core.pnl_gateway as G
    import core.sim as S
except Exception as e:
    st.error(f"❌ Falha ao importar módulos core: {e}")
    st.stop()

# ------------- Sidebar -------------
with st.sidebar:
    st.header("Configuração")
    valid = P.validate_project_structure()
    if not valid["current_parquet"]:
        st.error("Arquivo `data/parquet/current.parquet` não encontrado.")
    if not valid["parquet_dir"]:
        st.error("Pasta `data/parquet/` não encontrada.")
    if not valid["base_calculos_xlsx"]:
        st.warning("Planilha `data/premissas_pnl/base_calculos.xlsx` ausente (fallback RE será usado no YTG).")

    # Anos disponíveis
    try:
        curr = M.load_current_long()
        anos = sorted(pd.to_numeric(curr["ano"], errors="coerce").dropna().astype(int).unique().tolist())
    except Exception:
        anos = []
    year = st.selectbox("Ano", anos, index=len(anos) - 1 if anos else 0, disabled=not anos)

    scen_label = st.selectbox("Cenário", ["BP (FY)", "RE (FY)", "Realizado (YTD+YTG)"], index=2)
    scale_label = st.selectbox("Escala visual", ["1x", "1.000x", "1.000.000x"], index=0)

    use_sim = False
    if "Realizado" in scen_label:
        use_sim = st.toggle("Usar Projeção (YTG)", value=False, help="OFF→ Grupo 3 (RES). ON→ Grupo 4 (UI).")

# Injeção de estado esperado pelo gateway/sim
st.session_state[S._SCEN_LABEL_KEY] = scen_label
st.session_state[S._SCALE_LABEL_KEY] = scale_label
st.session_state[S._USE_SIM_KEY] = use_sim

# ---------- Utils ----------
def _scale_factor(scale: str) -> int:
    return {"1x": 1, "1.000x": 1_000, "1.000.000x": 1_000_000}[scale]

def _format_br(x, casas=2):
    try:
        val = float(x)
    except Exception:
        return x
    s = f"{val:,.{casas}f}"  # 1,234,567.89
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s

def _apply_scale_and_format(df: pd.DataFrame, scale: str, casas=2) -> pd.DataFrame:
    factor = _scale_factor(scale)
    num_cols = [c for c in df.columns if c != "Conta"]
    out = df.copy()
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0) / factor
        out[c] = out[c].map(lambda v: _format_br(v, casas=casas))
    return out

# ---------- Diagnóstico ----------
with st.expander("🔎 Diagnóstico", expanded=False):
    st.write("Estrutura:", {k: str(v) if isinstance(v, Path) else v for k, v in valid.items()})
    try:
        last_re = M.find_latest_re_scenario(year if year else None)
        st.write("Último RE detectado:", last_re)
    except Exception as e:
        st.warning(f"RE não detectado: {e}")
    try:
        cut = M.realized_cutoff_by_year("Realizado")
        st.write("Cutoff Realizado:", cut)
    except Exception as e:
        st.warning(f"Cutoff não calculado: {e}")

# ---------- Execução principal ----------
if not anos:
    st.stop()

try:
    df = G.get_pnl_for_current_settings(year=year, scenario_label=scen_label, use_projection=use_sim)

except Exception as e:
    st.error(f"❌ Erro ao montar o P&L: {e}")
    st.stop()

required_cols = ["Conta"] + M.MONTHS_PT + ["Total Ano"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Contrato inválido. Faltando colunas: {missing}")
    st.dataframe(df.head(50), use_container_width=True)
    st.stop()

# Quando toggle OFF (Grupo 3), se vier tudo zero/vazio, mostra alerta amigável
if "Realizado" in scen_label and not use_sim:
    if all(float(df[c].sum()) == 0.0 for c in M.MONTHS_PT + ["Total Ano"]):
        st.warning("Toggle OFF (Grupo 3) retornou zeros. Verifique se o `res_working.parquet` tem dados para o ano selecionado.")

st.subheader(f"P&L — {scen_label} • {year}")
st.caption("Escala e formatação aplicadas apenas para exibição visual (padrão brasileiro).")

df_fmt = _apply_scale_and_format(df, scale_label, casas=2)
st.dataframe(df_fmt, use_container_width=True, height=560)

# KPIs simples
with st.container():
    c1, c2, c3, c4 = st.columns(4)
    try:
        idx = df.set_index("Conta")
        receita = float(idx.loc["receita_bruta", "Total Ano"]) if "receita_bruta" in idx.index else 0.0
        rl = float(idx.loc["receita_liquida", "Total Ano"]) if "receita_liquida" in idx.index else 0.0
        ebitda = float(idx.loc["ebitda", "Total Ano"]) if "ebitda" in idx.index else 0.0
        vol = float(idx.loc["volume_uc", "Total Ano"]) if "volume_uc" in idx.index else 0.0
    except Exception:
        receita = rl = ebitda = vol = 0.0

    f = _scale_factor(scale_label)
    c1.metric("Receita Bruta (Total Ano)", _format_br(receita / f))
    c2.metric("Receita Líquida (Total Ano)", _format_br(rl / f))
    c3.metric("EBITDA (Total Ano)", _format_br(ebitda / f))
    c4.metric("Volume UC (Total Ano)", _format_br(vol / f))
