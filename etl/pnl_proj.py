# -*- coding: utf-8 -*-
"""
etl/pnl_proj.py
Visualização simples do P&L mensal a partir do parquet gerado pelo ETL.

- Lê data/current.parquet (campos: ano, mes, indicador_id, valor, cenario, cenario_label)
- Filtros: Cenário (literal), Ano
- Escala: 1x, 1.000x, 1.000.000x
- Exibe pivô por Linha P&L (Indicador) x Meses + Total
- Formatação contábil inteira (sem decimais), negativos entre parênteses
"""

from __future__ import annotations

import io
from pathlib import Path
import pandas as pd
import streamlit as st

# -------------------- Configurações --------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../simulador_dre_streamlit
PARQUET_PATH = PROJECT_ROOT / "data" / "current.parquet"

# Ordem de linhas no P&L (id técnico)
INDICATOR_ORDER = [
    "volume_sellout",
    "volume_uc",
    "receita_bruta",
    "convenio_impostos",
    "receita_liquida",
    "insumos",
    "custos_toll_packer",
    "concentrado_incentivos",
    "fretes_t1",
    "estrutura_industrial_variavel",
    "custos_variaveis",
    "margem_variavel",
    "estrutura_industrial_fixa",
    "margem_bruta",
    "custos_log_t1_reg",
    "despesas_secundarias_t2",
    "perdas",
    "margem_contribuicao",
    "dme",
    "margem_contribuicao_liquida",
    "ajuste_cota_fixa",
    "opex_leao_reg",
    "cluster_ccil",
    "chargeback",
    "resultado_antes_regionalizacao",
    "custo_regionalizacao",
    "resultado_operacional",
    "depreciacao",
    "ebitda",
]

# Rótulos humanos exibidos na primeira coluna
INDICATOR_LABELS = {
    "volume_sellout": "Volume Sell Out (UC)",
    "volume_uc": "Volume (UC)",
    "receita_bruta": "Receita Bruta",
    "convenio_impostos": "Convênio/Impostos",
    "receita_liquida": "Receita Líquida",
    "insumos": "Insumos",
    "custos_toll_packer": "Custos Toll Packer",
    "concentrado_incentivos": "Concentrado/Incentivo de Vendas",
    "fretes_t1": "Fretes T1",
    "estrutura_industrial_variavel": "Estrutura Industrial - Variável",
    "custos_variaveis": "Custos Variáveis",
    "margem_variavel": "Margem Variável",
    "estrutura_industrial_fixa": "Estrutura Industrial - Fixa Reg",
    "margem_bruta": "Margem Bruta",
    "custos_log_t1_reg": "Custos Log. T1 Reg",
    "despesas_secundarias_t2": "Despesas Secundárias (T2)",
    "perdas": "Perdas",
    "margem_contribuicao": "Margem Contribuição",
    "dme": "DME",
    "margem_contribuicao_liquida": "Margem Contribuição Líquida",
    "ajuste_cota_fixa": "Ajuste Cota Fixa",
    "opex_leao_reg": "Opex Leão Reg",
    "cluster_ccil": "Cluster CCIL",
    "chargeback": "Charge Back",
    "resultado_antes_regionalizacao": "R. Oper. antes Regionalização",
    "custo_regionalizacao": "Custo Regionalizacao",
    "resultado_operacional": "Resultado Operacional",
    "depreciacao": "Depreciacao",
    "ebitda": "EBITDA",
}

MES_MAP = {
    1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez",
}
MONTH_COLS = [MES_MAP[i] for i in range(1, 13)]

SCALE_OPTIONS = {"1x": 1, "1.000x": 1_000, "1.000.000x": 1_000_000}

# -------------------- Utilitários --------------------
@st.cache_data(show_spinner=False)
def load_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # tipagem mínima
    df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")
    df["mes"] = pd.to_numeric(df["mes"], errors="coerce").astype("Int64")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce").fillna(0.0)
    df["indicador_id"] = df["indicador_id"].astype(str)
    df["cenario"] = df["cenario"].astype(str).str.strip()
    df["cenario_label"] = df["cenario_label"].astype(str).str.strip()
    return df

def fmt_accounting_int(x, scale: float = 1.0) -> str:
    """Formata inteiro contábil pt-BR, negativo entre parênteses, sem casas decimais."""
    try:
        v = int(round(float(x) / float(scale)))
    except Exception:
        return "—"
    s = f"{abs(v):,}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"({s})" if v < 0 else s

def finalize_display_int(df_num: pd.DataFrame, escala: float, label_col: str = "Linha P&L") -> pd.DataFrame:
    """
    Recebe DF numérico com linhas = indicador_id e colunas = meses [+ Total].
    Retorna DF para exibição: primeira coluna com rótulo humano, meses/Total formatados inteiros.
    """
    df = df_num.copy()

    # Garante indicador_id como coluna
    if "indicador_id" not in df.columns and getattr(df.index, "name", None) == "indicador_id":
        df = df.reset_index()

    # Coluna de rótulo
    if label_col not in df.columns:
        df[label_col] = df["indicador_id"].map(INDICATOR_LABELS).fillna(df["indicador_id"])

    # Ordena linhas pela ordem definida (quando possível)
    order_map = {k: i for i, k in enumerate(INDICATOR_ORDER)}
    df["_ord"] = df["indicador_id"].map(order_map)
    df = df.sort_values(["_ord", label_col]).drop(columns=["_ord"])

    # Reordena colunas: label + meses + Total (se houver)
    ordered = [label_col] + [c for c in MONTH_COLS if c in df.columns]
    if "Total" in df.columns:
        ordered += ["Total"]
    df = df[[c for c in ordered if c in df.columns]].copy()

    # Formata apenas numéricas
    for c in df.columns:
        if c == label_col:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).apply(lambda v: fmt_accounting_int(v, escala))

    return df

def build_pivot(df: pd.DataFrame, ano: int, cenario_literal: str) -> pd.DataFrame:
    """Gera pivô por indicador x meses (numérico), mais Total (numérico)."""
    dfx = df[(df["ano"] == ano) & (df["cenario"] == cenario_literal)].copy()
    if dfx.empty:
        return pd.DataFrame(columns=["indicador_id"] + MONTH_COLS + ["Total"])

    # soma valores por indicador_id x mes
    agg = dfx.groupby(["indicador_id", "mes"], as_index=False)["valor"].sum()
    agg["mes_nome"] = agg["mes"].map(MES_MAP)

    pv = agg.pivot(index="indicador_id", columns="mes_nome", values="valor").reindex(columns=MONTH_COLS, fill_value=0.0)
    pv = pv.reset_index()

    # Total numérico
    pv["Total"] = pv[MONTH_COLS].sum(axis=1)

    return pv

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode()

# -------------------- App --------------------
def main():
    st.set_page_config(page_title="P&L (Parquet)", page_icon="📊", layout="wide")
    st.title("📊 P&L – Visualização por Mês (Parquet)")

    # Carrega base
    if not PARQUET_PATH.exists():
        st.error(f"Parquet não encontrado em: {PARQUET_PATH}")
        st.stop()

    df_all = load_parquet(PARQUET_PATH)

    # Filtros (Topo)
    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        cenarios = sorted(df_all["cenario"].dropna().unique().tolist())
        if not cenarios:
            st.error("Nenhum cenário disponível no parquet.")
            st.stop()
        cenario_sel = st.selectbox("Cenário (literal do Excel)", options=cenarios, index=0)

    with col2:
        anos = sorted(df_all.loc[df_all["cenario"] == cenario_sel, "ano"].dropna().unique().tolist())
        ano_sel = st.selectbox("Ano", options=anos, index=len(anos) - 1)

    with col3:
        escala_label = st.selectbox("Escala", options=list(SCALE_OPTIONS.keys()), index=0)
    escala = SCALE_OPTIONS[escala_label]

    # Gera pivô numérico
    pv = build_pivot(df_all, ano_sel, cenario_sel)

    if pv.empty:
        st.warning("Não há dados para o filtro selecionado.")
        st.stop()

    # Exibição (formatada)
    show_df = finalize_display_int(pv, escala=escala, label_col="Linha P&L")
    st.dataframe(show_df, use_container_width=True, height=560)

    # Download CSV (numérico escalado e inteiro)
    export = pv.copy()
    num_cols = [c for c in MONTH_COLS + ["Total"] if c in export.columns]
    export[num_cols] = (export[num_cols] / escala).round().astype("Int64")
    export.insert(1, "Linha P&L", export["indicador_id"].map(INDICATOR_LABELS).fillna(export["indicador_id"]))
    export = export.drop(columns=["indicador_id"])

    csv_bytes = df_to_csv_bytes(export)
    st.download_button(
        "⬇️ Baixar CSV (inteiros, escala aplicada)",
        data=csv_bytes,
        file_name=f"pnl_{cenario_sel}_{ano_sel}.csv",
        mime="text/csv",
        use_container_width=False,
    )

    # Rodapé
    st.caption(f"Fonte: {PARQUET_PATH}")

if __name__ == "__main__":
    main()
