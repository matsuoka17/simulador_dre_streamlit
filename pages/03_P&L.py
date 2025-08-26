# pages/03_DRE.py
# --------------------------------------------------------------------------------------
# DRE (Parquet) — 3 abas no MESMO arquivo:
#   1) P&L Mensal (com UC por mês, UC (Visão), UC (Ano) e Total Visão quando subset)
#   2) BP25 vs FY (YTD+YTG) — 5 colunas (Linha, BP, FY, Δ Abs, Δ %), sem barra de rolagem (autoHeight)
#   3) RES vs UI — 5 colunas (Linha, RES, UI, Δ Abs, Δ %), comparando RES_WORKING vs UI
# --------------------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List, Tuple
from io import BytesIO
import re
import sys

import numpy as np
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.shared import JsCode

from core.fs_utils import read_parquet_first_found, debug_parquet_status

# ⬇️ SEMPRE antes do primeiro st.*
st.set_page_config(page_title="DRE", page_icon="📊", layout="wide")

# Diagnóstico foi movido para o rodapé (não chamar aqui)
# debug_parquet_status()

df_current = read_parquet_first_found([
    "data/parquet/current.parquet",
    "data/current.parquet",
])
df_res = read_parquet_first_found([
    "data/parquet/res_working.parquet",
    "data/res_working.parquet",
])

# ===== Imports do projeto =====
from core.models import (
    load_current_long,
    dre_matrix_total,
    dre_matrix_by_family,
    MONTH_MAP_PT as MONTHS_PT,
    # RES_WORKING (baseline YTG)
    res_volume_by_family_long,
    res_volume_total_by_month,
)

# --- Imports robustos para calculator e sim ---
try:
    from core.calculator import build_simulado_pivot  # type: ignore
except (ImportError, ModuleNotFoundError):
    try:
        from simulador_dre_streamlit.core.calculator import build_simulado_pivot  # type: ignore
    except (ImportError, ModuleNotFoundError):
        _root = Path(__file__).resolve().parents[1]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        from core.calculator import build_simulado_pivot  # type: ignore

try:
    import core.sim as S  # type: ignore
except (ImportError, ModuleNotFoundError):
    try:
        import simulador_dre_streamlit.core.sim as S  # type: ignore
    except (ImportError, ModuleNotFoundError):
        _root = Path(__file__).resolve().parents[1]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        import core.sim as S  # type: ignore

# ---------------- Helpers locais para volumes UI (wide -> long) ----------------
MONTHS_LOWER: Dict[str, Tuple[int, str]] = {name.lower(): (m, name) for m, name in MONTHS_PT.items()}
FAM_COL_CAND = ["Família Comercial", "Familia Comercial", "familia_comercial", "Familia"]


def _fam_col_local(df: pd.DataFrame) -> str:
    for c in FAM_COL_CAND:
        if c in df.columns:
            return c
    return "Família Comercial" if "Família Comercial" in df.columns else df.columns[0]


def get_volumes_from_ui_long(year: int) -> Optional[pd.DataFrame]:
    """
    Converte st.session_state['volumes_wide'][year] (Família x Meses) para formato long:
    colunas: [ano, mes, 'Família Comercial', volume].
    Se não existir 'volumes_wide', tenta st.session_state['volumes_edit'].
    Retorna None se não houver dados para o ano.
    """
    vw = st.session_state.get("volumes_wide")
    if isinstance(vw, dict) and year in vw and isinstance(vw[year], pd.DataFrame):
        dfw = vw[year].copy()
        fam_col = _fam_col_local(dfw)
        month_idx: Dict[int, str] = {}
        for c in dfw.columns:
            lc = str(c).strip().lower()
            if lc in MONTHS_LOWER:
                month_idx[MONTHS_LOWER[lc][0]] = c
        rows = []
        for m in range(1, 13):
            col = month_idx.get(m)
            vals = pd.to_numeric(dfw[col], errors="coerce").fillna(0.0) if col is not None else 0.0
            row = pd.DataFrame({
                "ano": year,
                "mes": m,
                "Família Comercial": dfw[fam_col].astype(str),
                "volume": vals if hasattr(vals, "values") else [vals] * len(dfw)
            })
            rows.append(row)
        return pd.concat(rows, ignore_index=True)
    ve = st.session_state.get("volumes_edit")
    if isinstance(ve, pd.DataFrame) and {"ano", "mes", "volume"}.issubset(ve.columns):
        sub = ve[ve["ano"] == year].copy()
        return sub if not sub.empty else None
    return None


PARQUET_PATH = Path("data/parquet/current.parquet")
RES_WORKING_PATH = Path("data/parquet/res_working.parquet")
BASE_CALC_PATH = Path("data/premissas_pnl/base_calculos.xlsx")

VOLUME_ID = "volume_uc"

ROW_SPECS: List[Tuple[str, str]] = [
    ("volume_uc", "volume_uc"),
    ("receita_bruta", "receita_bruta"),
    ("convenio_impostos", "convenio_impostos"),
    ("receita_liquida", "receita_liquida"),
    ("insumos", "insumos"),
    ("custos_toll_packer", "custos_toll_packer"),
    ("fretes_t1", "fretes_t1"),
    ("estrutura_industrial_variavel", "estrutura_industrial_variavel"),
    ("custos_variaveis", "custos_variaveis"),
    ("margem_variavel", "margem_variavel"),
    ("estrutura_industrial_fixa", "estrutura_industrial_fixa"),
    ("margem_bruta", "margem_bruta"),
    ("custos_log_t1_reg", "custos_log_t1_reg"),
    ("despesas_secundarias_t2", "despesas_secundarias_t2"),
    ("perdas", "perdas"),
    ("margem_contribuicao", "margem_contribuicao"),
    ("dme", "dme"),
    ("margem_contribuicao_liquida", "margem_contribuicao_liquida"),
    ("opex_leao_reg", "opex_leao_reg"),
    ("chargeback", "chargeback"),
    ("resultado_operacional", "resultado_operacional"),
    ("depreciacao", "depreciacao"),
    ("ebitda", "ebitda"),
]

DISPLAY_LABELS: Dict[str, str] = {
    "volume_uc": "Volume (UC)",
    "receita_bruta": "Receita Bruta",
    "convenio_impostos": "Convênio / Impostos",
    "receita_liquida": "Receita Líquida",
    "insumos": "Insumos",
    "custos_toll_packer": "Custos Toll Packer",
    "fretes_t1": "Fretes T1",
    "estrutura_industrial_variavel": "Estrutura Industrial Variável",
    "custos_variaveis": "Custos Variáveis",
    "margem_variavel": "Margem Variável",
    "estrutura_industrial_fixa": "Estrutura Industrial Fixa",
    "margem_bruta": "Margem Bruta",
    "custos_log_t1_reg": "Custos Log T1 Reg",
    "despesas_secundarias_t2": "Despesas Secundárias T2",
    "perdas": "Perdas",
    "margem_contribuicao": "Margem de Contribuição",
    "dme": "DME",
    "margem_contribuicao_liquida": "Margem de Contribuição Líquida",
    "opex_leao_reg": "Opex Leão Reg",
    "chargeback": "Charge Back",
    "resultado_operacional": "Resultado Operacional",
    "depreciacao": "Depreciação",
    "ebitda": "EBITDA",
}

BOLD_GREY_ROWS = {
    "volume_uc",
    "receita_liquida",
    "custos_variaveis",
    "margem_variavel",
    "margem_bruta",
    "margem_contribuicao",
    "margem_contribuicao_liquida",
    "resultado_operacional",
    "ebitda",
}


# --------------------------------------------------------------------------------------
# Helpers de cenário / carga
# --------------------------------------------------------------------------------------
def is_realizado(s: str) -> bool:
    return isinstance(s, str) and ("realiz" in s.lower())


def is_bp(s: str) -> bool:
    return isinstance(s, str) and s.lower().replace(" ", "").startswith("bp")


@st.cache_data(show_spinner=False)
def load_parquet(_: Path) -> pd.DataFrame:
    df = load_current_long()
    for c in ("ano", "mes"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    if "valor" in df.columns:
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce").fillna(0.0)
    if "indicador_id" in df.columns:
        df["indicador_id"] = df["indicador_id"].astype(str)
    if "cenario" in df.columns:
        df["cenario"] = df["cenario"].astype(str).str.strip()
    fam_cols = [c for c in df.columns if str(c).lower().startswith("fam")]
    if fam_cols and "Família Comercial" not in df.columns:
        df = df.rename(columns={fam_cols[0]: "Família Comercial"})
    return df


def list_years(df: pd.DataFrame) -> List[int]:
    return sorted([int(x) for x in df["ano"].dropna().unique()])


def scenarios_for_year(df: pd.DataFrame, year: int) -> List[str]:
    sc = df.loc[df["ano"] == year, "cenario"].dropna().astype(str).unique().tolist()
    sc = sorted(sc, key=lambda s: (0 if is_bp(s) else (1 if is_realizado(s) else 2), s))
    return sc


def scenario_realizado_for_year(df: pd.DataFrame, year: int) -> Optional[str]:
    for s in scenarios_for_year(df, year):
        if is_realizado(s):
            return s
    return None


def cutoff_from_realizado(df: pd.DataFrame, year: int) -> int:
    scen = scenario_realizado_for_year(df, year)
    if scen is None:
        return 0
    m = (df[(df["ano"] == year) & (df["cenario"] == scen) & (df["indicador_id"] == VOLUME_ID)]
         .groupby("mes")["valor"].sum())
    if m.empty:
        return 0
    nonzero = m[m != 0]
    return int(nonzero.index.max()) if not nonzero.empty else 0


def find_baseline_scenario(df: pd.DataFrame, year: int) -> Optional[str]:
    for s in scenarios_for_year(df, year):
        if is_bp(s):
            return s
    for s in scenarios_for_year(df, year):
        if s.lower().replace(" ", "").startswith("re"):
            return s
    return None


# ---------- Pivot no formato do P&L (linhas fixas de ROW_SPECS) ----------
def pivot_pnl_alias(df: pd.DataFrame, year: int, scenario: str) -> pd.DataFrame:
    month_cols = [MONTHS_PT[m] for m in range(1, 13)]
    base = df[(df["ano"] == year) & (df["cenario"] == scenario)].copy()

    if base.empty:
        rows = []
        for display_id, _src in ROW_SPECS:
            rows.append({
                "indicador_id": display_id,
                "Indicador": DISPLAY_LABELS.get(display_id, display_id),
                **{mc: 0 for mc in month_cols},
                "Total Ano": 0
            })
        return pd.DataFrame(rows)

    agg = (base.groupby(["indicador_id", "mes"], as_index=False)["valor"].sum()
           .pivot(index="indicador_id", columns="mes", values="valor")
           .reindex(columns=range(1, 13)).fillna(0.0))

    rows = []
    for display_id, source_id in ROW_SPECS:
        if source_id in agg.index:
            vals = {MONTHS_PT[m]: float(agg.loc[source_id, m]) for m in range(1, 13)}
        else:
            vals = {MONTHS_PT[m]: 0.0 for m in range(1, 13)}
        rows.append({
            "indicador_id": display_id,
            "Indicador": DISPLAY_LABELS.get(display_id, display_id),
            **{k: int(np.rint(v)) for k, v in vals.items()},
            "Total Ano": int(np.rint(sum(vals.values())))
        })

    out = pd.DataFrame(rows)
    return out[["indicador_id", "Indicador", *month_cols, "Total Ano"]]


# ---------- UI Volumes (TOTAL MENSAL) ----------
def _sum_by_month_from_volumes_wide(dfw: pd.DataFrame) -> Dict[int, int]:
    col_map: Dict[int, str] = {}
    for c in dfw.columns:
        lc = str(c).strip().lower()
        if lc in MONTHS_LOWER:
            col_map[MONTHS_LOWER[lc][0]] = c
    totals: Dict[int, int] = {}
    for m in range(1, 13):
        col = col_map.get(m)
        if col is None:
            totals[m] = 0
        else:
            totals[m] = int(pd.to_numeric(dfw[col], errors="coerce").fillna(0.0).sum())
    return totals


def get_volumes_from_ui_total(year: int) -> Optional[Dict[int, int]]:
    vw = st.session_state.get("volumes_wide")
    if isinstance(vw, dict) and year in vw and isinstance(vw[year], pd.DataFrame):
        return _sum_by_month_from_volumes_wide(vw[year])
    ve = st.session_state.get("volumes_edit")
    if isinstance(ve, pd.DataFrame) and {"ano", "mes", "volume"}.issubset(ve.columns):
        sub = ve[ve["ano"] == year]
        if not sub.empty:
            g = pd.to_numeric(sub["volume"], errors="coerce").fillna(0.0).groupby(sub["mes"]).sum()
            return {int(m): int(g.get(m, 0.0)) for m in range(1, 13)}
    return None


def get_volumes_from_res(year: int) -> Optional[Dict[int, int]]:
    d = res_volume_total_by_month(year)
    if not d:
        return None
    return {int(k): int(v) for k, v in d.items()}


# --------------------------------------------------------------------------------------
# Helper para RES Working pivot
# --------------------------------------------------------------------------------------
def build_res_working_pivot(df: pd.DataFrame, year: int, cutoff: int) -> pd.DataFrame:
    """
    Constrói pivot usando dados do RES_WORKING para comparação com UI.
    Combina YTD (realizado) + YTG (res_working).
    """
    # Busca cenário realizado para YTD
    scen_real = scenario_realizado_for_year(df, year)
    if not scen_real:
        return pivot_pnl_alias(df, year, find_baseline_scenario(df, year) or "")

    # Dados do RES_WORKING para YTG
    try:
        volumes_res_df = res_volume_by_family_long()
        res_totals = get_volumes_from_res(year)
    except:
        volumes_res_df = pd.DataFrame()
        res_totals = None

    # Constrói pivot usando RES_WORKING
    piv_real = pivot_pnl_alias(df, year, scen_real)

    if not volumes_res_df.empty and res_totals:
        res_piv = build_simulado_pivot(
            df=df,
            piv_real=piv_real,
            year=year,
            cutoff=cutoff,
            base_calc_path=BASE_CALC_PATH,
            volumes_edit=None,  # Não usa UI
            volumes_res=volumes_res_df,  # Usa RES_WORKING
            volume_mode="res",
            dme_pct=0.102,
            ui_month_totals=res_totals,
            conv_source="excel",
        )
        return res_piv

    return piv_real


# --------------------------------------------------------------------------------------
# Estilo / exports
# --------------------------------------------------------------------------------------
def make_value_formatter(scale: int) -> JsCode:
    return JsCode(
        f"""
        function(params) {{
            var v = params.value;
            if (v === null || v === undefined) return "-";
            var scaled = Math.round(v / {scale});
            return scaled.toString().replace(/\\B(?=(\\d{{3}})+(?!\\d))/g, ".");
        }}
        """
    )


def make_row_style_js() -> JsCode:
    anchors = list(BOLD_GREY_ROWS)
    js_list = "[" + ",".join([f"'{x}'" for x in anchors]) + "]"
    return JsCode(
        f"""
        function(params){{
          if (params.data && params.data.indicador_id === 'resultado_operacional'){{
            return {{'backgroundColor':'#FFF8C5','fontWeight':'700'}};
          }}
          if ({js_list}.includes(params.data.indicador_id)){{
            return {{'backgroundColor':'F3F4F6','fontWeight':'600'}};
          }}
          return null;
        }}
        """
    )


def on_fit_events_js() -> Tuple[JsCode, JsCode]:
    on_ready = JsCode("function(params){ params.api.sizeColumnsToFit(); }")
    on_col_vis = JsCode("function(params){ params.api.sizeColumnsToFit(); }")
    return on_ready, on_col_vis


def _short_title(s: str) -> str:
    up = s.upper().replace(" ", "")
    if up.startswith("BP"): return "BP"
    if up.startswith("RE"): return "RE"
    if "REALIZ" in up: return "Realizado"
    return s


def make_excel_bytes(df_export: pd.DataFrame, sheet_name: str = "DRE") -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df_export.to_excel(writer, index=False, sheet_name=sheet_name)
    bio.seek(0)
    return bio.getvalue()


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------
def main():
    if not PARQUET_PATH.exists():
        st.error(f"❌ Parquet não encontrado: {PARQUET_PATH}")
        st.stop()

    # ---------------- Sidebar: Controles Globais ----------------
    with st.sidebar:
        st.header("⚙️ Controles")
        st.session_state.setdefault(S._USE_SIM_KEY, True)

        st.toggle(
            "Usar Projeção (UI) para YTG",
            key=S._USE_SIM_KEY,
            value=st.session_state[S._USE_SIM_KEY],
        )
        use_sim = st.session_state[S._USE_SIM_KEY]

        escala_label = st.selectbox(
            "Escala",
            ["1x", "1.000x", "1.000.000x"],
            index=["1x", "1.000x", "1.000.000x"].index(st.session_state.get(S._SCALE_LABEL_KEY, "1.000x"))
        )
        st.session_state[S._SCALE_LABEL_KEY] = escala_label
        st.session_state["dre_scale_label"] = escala_label

        st.caption("• YTD vem do cenário Realizado do CURRENT.\n• YTG usa UI se marcado; caso contrário, RES_WORKING.")

        st.divider()
        st.header("📁 Status dos Arquivos (Sidebar)")
        try:
            df_chk = load_current_long()
            ok_current = not df_chk.empty
        except Exception:
            ok_current = False
        st.write("current.parquet:", "✅" if ok_current else "❌")

        try:
            res_df_chk = res_volume_by_family_long()
            ok_res = not res_df_chk.empty
        except Exception:
            ok_res = False
        st.write("res_working.parquet:", "✅" if ok_res else "❌")

    # ---------------- Base CURRENT ----------------
    df = load_parquet(PARQUET_PATH)
    years = list_years(df)
    default_year = years[-1] if years else pd.Timestamp.today().year

    st.session_state.setdefault("dre_year", default_year)

    st.markdown("### Demonstração de Resultado (DRE) — Comparativo P&L Mensal")

    # Controles COMUNS às abas
    year = st.session_state.get("dre_year") or default_year
    escala_label = st.session_state.get(S._SCALE_LABEL_KEY, "1.000x")
    scale = {"1x": 1, "1.000x": 1_000, "1.000.000x": 1_000_000}[escala_label]

    # Cenários
    scens = scenarios_for_year(df, year)
    if not scens:
        st.warning("Nenhum cenário para o ano selecionado.")
        st.stop()

    scen_real = scenario_realizado_for_year(df, year)
    cutoff = cutoff_from_realizado(df, year)
    baseline_name = find_baseline_scenario(df, year)

    # --------- PRÉ-CÁLCULOS: volumes conforme TOGGLE ---------
    use_sim = bool(st.session_state.get(S._USE_SIM_KEY, True))
    volumes_ui_long = get_volumes_from_ui_long(year) if use_sim else None
    volumes_res_df = res_volume_by_family_long() if (not use_sim) else pd.DataFrame()
    ui_totals = get_volumes_from_ui_total(year) if use_sim else None
    volume_mode = "ui" if use_sim else "res"

    piv_real = pivot_pnl_alias(df, year, scen_real) if scen_real else None
    piv_bp = pivot_pnl_alias(df, year, baseline_name) if baseline_name else None

    # ----------------- ABAS -----------------
    tab1, tab2, tab3 = st.tabs(["P&L Mensal", "BP25 vs Projeção", "Passo 2 vs Projeção"])

    # ==================================================================================
    # ABA 1 — P&L MENSAL
    # ==================================================================================
    with tab1:
        scen_sel = st.selectbox(
            "Cenário para comparativo",
            scens,
            index=min(1, len(scens) - 1) if len(scens) > 1 else 0,
            key="dre_scen_only"
        )
        piv_sel = pivot_pnl_alias(df, year, scen_sel)

        def sum_row(piv: pd.DataFrame, rid: str) -> int:
            row = piv.loc[piv["indicador_id"] == rid]
            return int(row["Total Ano"].values[0]) if not row.empty else 0

        c1, c2, c3 = st.columns(3)

        def kpi_card(label: str, sim_val: float, scen_val: float):
            a = int(round(sim_val))
            b = int(round(scen_val))
            diff = a - b
            pct = (diff / b * 100.0) if b != 0 else (0.0 if a == 0 else 100.0)
            st.metric(
                label,
                value=f"{(a // scale):,}".replace(",", "."),
                delta=f"{(diff // scale):,}".replace(",", ".") + f"  ({pct:+.1f}%)",
                help=f"Simulado: {a:,}  |  Cenário: {b:,}".replace(",", "."),
            )

        # Simulado para KPIs (usa toggle global)
        if piv_real is not None:
            piv_table = build_simulado_pivot(
                df=df,
                piv_real=piv_real,
                year=year,
                cutoff=cutoff,
                base_calc_path=BASE_CALC_PATH,
                volumes_edit=volumes_ui_long,
                volumes_res=volumes_res_df,
                volume_mode=volume_mode,
                dme_pct=0.102,
                ui_month_totals=ui_totals,
                conv_source="excel",
            )
        else:
            piv_table = None

        with c1:
            st.markdown('<div class="sim-badge">Simulado</div>', unsafe_allow_html=True)
            kpi_card("Volume (UC)",
                     sum_row(piv_table, "volume_uc") if piv_table is not None else 0,
                     sum_row(piv_sel, "volume_uc"))
        with c2:
            kpi_card("Receita Líquida",
                     sum_row(piv_table, "receita_liquida") if piv_table is not None else 0,
                     sum_row(piv_sel, "receita_liquida"))
        with c3:
            kpi_card("Resultado Operacional",
                     sum_row(piv_table, "resultado_operacional") if piv_table is not None else 0,
                     sum_row(piv_sel, "resultado_operacional"))

        st.markdown("""
        <style>
        .sim-badge{background:#0ea5e9;color:white;padding:4px 8px;border-radius:8px;
                   font-size:12px;text-align:center;width:80px}
        .ag-right-aligned-cell{ text-align:right !important; font-variant-numeric: tabular-nums; }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Controles da tabela (sem on_change/rerun)
        ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.2, 1.0, 1.0, 1.6], gap="small")
        with ctrl1:
            st.checkbox(
                "Detalhado",
                key="show_all_rows_t1",
                value=bool(st.session_state.get("show_all_rows_t1", False)),
                help="Exibir todas as linhas."
            )
        with ctrl2:
            st.checkbox("YTD", key="show_ytd_detail_t1")
        with ctrl3:
            st.checkbox("YTG", key="show_ytg_detail_t1")

        default_table_scen = scenario_realizado_for_year(df, year) or scens[0]
        st.selectbox(
            "Cenário Base da Tabela",
            scens,
            index=scens.index(default_table_scen) if default_table_scen in scens else 0,
            key="dre_table_scen"
        )

        # Projeção na tabela (quando base = Realizado)
        piv_table = pivot_pnl_alias(df, year, st.session_state["dre_table_scen"])
        if is_realizado(st.session_state["dre_table_scen"]) and piv_real is not None:
            piv_table = build_simulado_pivot(
                df=df, piv_real=piv_table, year=year, cutoff=cutoff,
                base_calc_path=BASE_CALC_PATH,
                volumes_edit=volumes_ui_long,
                volumes_res=volumes_res_df,
                volume_mode=volume_mode,
                dme_pct=0.102,
                ui_month_totals=ui_totals,
                conv_source="excel",
            )

        # --- Visão (todas x principais): só decidir UMA vez ---
        show_all = bool(st.session_state.get("show_all_rows_t1", False))
        if show_all:
            piv_table_view = piv_table.copy()
        else:
            keep_ids = set(BOLD_GREY_ROWS)
            piv_table_view = (
                piv_table[piv_table["indicador_id"].isin(keep_ids)]
                .reset_index(drop=True)
                .copy()
            )

        # ---- Cálculos auxiliares para UC e Totais (NÃO reatribuir piv_table_view) ----
        vol_por_mes = {MONTHS_PT[m]: int(piv_table.loc[piv_table["indicador_id"] == "volume_uc", MONTHS_PT[m]].sum())
                       for m in range(1, 13)}
        vol_total_ano = int(piv_table.loc[piv_table["indicador_id"] == "volume_uc", "Total Ano"].sum())

        # Determina meses "visíveis" conforme toggles YTD/YTG (para Total Visão)
        show_ytd = bool(st.session_state.get("show_ytd_detail_t1", False))
        show_ytg = bool(st.session_state.get("show_ytg_detail_t1", False))
        meses_visiveis: List[str] = []
        cutoff_safe = int(cutoff) if cutoff else 0
        if show_ytd and cutoff_safe > 0:
            meses_visiveis += [MONTHS_PT[m] for m in range(1, cutoff_safe + 1)]
        if show_ytg and cutoff_safe < 12:
            meses_visiveis += [MONTHS_PT[m] for m in range(cutoff_safe + 1, 13)]
        seen = set()
        meses_visiveis = [x for x in meses_visiveis if not (x in seen or seen.add(x))]
        vol_total_visao = int(sum(vol_por_mes.get(mc, 0) for mc in meses_visiveis)) if meses_visiveis else 0

        meses_todos = [MONTHS_PT[m] for m in range(1, 13)]
        is_subset = bool(meses_visiveis) and set(meses_visiveis) != set(meses_todos)
        if is_subset:
            piv_table_view["Total Visão"] = piv_table_view[meses_visiveis].sum(axis=1).astype(float)
            if vol_total_visao > 0:
                piv_table_view["UC (Visão)"] = (
                            piv_table_view["Total Visão"].astype(float) / float(vol_total_visao)).fillna(0.0)
            else:
                piv_table_view["UC (Visão)"] = 0.0

        # UC por mês
        for mc in meses_todos:
            vol_m = vol_por_mes.get(mc, 0)
            uc_col = f"UC {mc}"
            if vol_m > 0:
                piv_table_view[uc_col] = (piv_table_view[mc].astype(float) / float(vol_m)).fillna(0.0)
            else:
                piv_table_view[uc_col] = 0.0

        # UC (Ano)
        if vol_total_ano > 0:
            piv_table_view["UC (Ano)"] = (piv_table_view["Total Ano"].astype(float) / float(vol_total_ano)).fillna(0.0)
        else:
            piv_table_view["UC (Ano)"] = 0.0

        # Colunas agrupadas YTD / YTG
        ytd_cols = [v for k, v in MONTHS_PT.items() if k <= cutoff_safe]
        ytg_cols = [v for k, v in MONTHS_PT.items() if k > cutoff_safe]

        fmt = make_value_formatter(scale)
        row_style = make_row_style_js()
        on_ready, on_col_vis = on_fit_events_js()

        fmt_uc = JsCode("""
function(params){
  var v = params.value;
  if (v === null || v === undefined) return "-";
  var n = Number(v);
  if (isNaN(n)) return "-";
  return n.toLocaleString('pt-BR', {minimumFractionDigits: 2, maximumFractionDigits: 2});
}
""")

        def child(col_name: str, hidden: bool = False):
            return {
                "field": col_name,
                "editable": False,
                "type": "numericColumn",
                "valueFormatter": fmt,
                "hide": hidden,
                "cellClass": "ag-right-aligned-cell",
            }

        def child_uc(col_name: str, hidden: bool = False):
            return {
                "field": col_name,
                "editable": False,
                "type": "numericColumn",
                "valueFormatter": fmt_uc,
                "hide": hidden,
                "cellClass": "ag-right-aligned-cell",
            }

        ytd_children = []
        for c in ytd_cols:
            ytd_children.append(child(c, hidden=(not show_ytd)))
            ytd_children.append(child_uc(f"UC {c}", hidden=(not show_ytd)))

        ytg_children = []
        for c in ytg_cols:
            ytg_children.append(child(c, hidden=(not show_ytg)))
            ytg_children.append(child_uc(f"UC {c}", hidden=(not show_ytg)))

        column_defs = [
            {"field": "indicador_id", "hide": True},
            {"headerName": "Indicador", "field": "Indicador", "pinned": "left", "editable": False},
        ]
        if ytd_children:
            column_defs.append({
                "headerName": f"YTD (<= M{cutoff_safe:02d})" if cutoff_safe > 0 else "YTD",
                "marryChildren": True, "children": ytd_children,
            })
        if ytg_children:
            column_defs.append({
                "headerName": f"YTG (>{cutoff_safe:02d})" if cutoff_safe > 0 else "YTG",
                "marryChildren": True, "children": ytg_children,
            })

        existing = {c["field"] for c in ytd_children + ytg_children}
        for m in range(1, 13):
            mc = MONTHS_PT[m]
            if mc not in existing:
                column_defs.append(child(mc, hidden=True))
                column_defs.append(child_uc(f"UC {mc}", hidden=True))

        if "Total Visão" in piv_table_view.columns:
            column_defs.append(child("Total Visão", hidden=False))
            column_defs.append(child_uc("UC (Visão)", hidden=False))
        column_defs.append(child("Total Ano", hidden=False))
        column_defs.append(child_uc("UC (Ano)", hidden=False))

        grid_options = {
            "columnDefs": column_defs,
            "defaultColDef": {"resizable": True, "sortable": False, "filter": False,
                              "cellStyle": {"fontVariantNumeric": "tabular-nums"}},
            "getRowStyle": row_style,
            "rowHeight": 34, "suppressMovableColumns": True, "ensureDomOrder": True,
            "onFirstDataRendered": on_ready, "onColumnVisible": on_col_vis,
            "domLayout": "autoHeight",
        }

        scen_slug = re.sub(r'[^A-Za-z0-9]+', '-', str(_short_title(st.session_state["dre_table_scen"])))
        grid_key = f"dre_grid_{year}_{scen_slug}_{int(st.session_state.get('show_ytd_detail_t1', False))}_{int(use_sim)}_{int(st.session_state.get('show_all_rows_t1', False))}_{int(st.session_state.get('show_ytg_detail_t1', False))}_{scale}"

        AgGrid(
            piv_table_view,
            gridOptions=grid_options,
            data_return_mode="AS_INPUT",
            update_mode=GridUpdateMode.NO_UPDATE,
            allow_unsafe_jscode=True,
            theme="balham",
            fit_columns_on_grid_load=True,
            key=grid_key,
        )

    # ==================================================================================
    # ABA 2 — BP25 vs FY (YTD+YTG) | 5 colunas
    # ==================================================================================
    with tab2:
        st.markdown("#### BP25 vs Projeção")

        colf1, _ = st.columns([2, 1])
        with colf1:
            st.checkbox("Detalhado", key="show_all_rows_t2", help="Exibir todas as linhas.")

        bp_piv = piv_bp if piv_bp is not None else pivot_pnl_alias(df, year, baseline_name or scens[0])

        if piv_real is not None:
            fy_piv = build_simulado_pivot(
                df=df,
                piv_real=piv_real,
                year=year,
                cutoff=cutoff,
                base_calc_path=BASE_CALC_PATH,
                volumes_edit=(volumes_ui_long if use_sim else None),
                volumes_res=(res_volume_by_family_long() if (not use_sim) else pd.DataFrame()),
                volume_mode=("ui" if use_sim else "res"),
                dme_pct=0.102,
                ui_month_totals=(ui_totals if use_sim else None),
                conv_source="excel",
            )
        else:
            fy_piv = pivot_pnl_alias(df, year, baseline_name or scens[0])

        req_cols = ['indicador_id', 'Indicador', 'Total Ano']
        bp_view = bp_piv[req_cols].rename(columns={'Total Ano': 'BP25 (FY)'}).copy()
        fy_view = fy_piv[req_cols].rename(columns={'Total Ano': 'FY (YTD+YTG)'}).copy()

        order_ids = list(bp_view['indicador_id'])
        df5 = bp_view.merge(fy_view[['indicador_id', 'FY (YTD+YTG)']], on='indicador_id', how='left')
        extras = fy_view[~fy_view['indicador_id'].isin(df5['indicador_id'])]
        if not extras.empty:
            df5 = pd.concat([df5, extras], ignore_index=True, sort=False)
            for x in extras['indicador_id'].tolist():
                if x not in order_ids:
                    order_ids.append(x)

        df5['__ord__'] = pd.Categorical(df5['indicador_id'], categories=order_ids, ordered=True)
        df5 = df5.sort_values('__ord__').drop(columns='__ord__')

        if not st.session_state["show_all_rows_t2"]:
            keep_ids = set(BOLD_GREY_ROWS)
            df5 = df5[df5['indicador_id'].isin(keep_ids)].reset_index(drop=True)

        df5['BP25 (FY)'] = df5['BP25 (FY)'].fillna(0.0)
        df5['FY (YTD+YTG)'] = df5['FY (YTD+YTG)'].fillna(0.0)
        df5['Δ Abs'] = df5['FY (YTD+YTG)'] - df5['BP25 (FY)']
        bp_safe = df5['BP25 (FY)'].replace({0: np.nan})
        df5['Δ %'] = ((df5['FY (YTD+YTG)'] - df5['BP25 (FY)']) / bp_safe) * 100.0

        # -------- Tabela estilo AgGrid (sem rolagem) --------
        escala_label = st.session_state.get(S._SCALE_LABEL_KEY, "1.000x")
        scale = {"1x": 1, "1.000x": 1_000, "1.000.000x": 1_000_000}[escala_label]
        fmt = make_value_formatter(scale)

        fmt_pct = JsCode("""
function(params){
  var v = params.value;
  if (v === null || v === undefined) return "-";
  var n = Number(v);
  if (isNaN(n)) return "-";
  var s = (n >= 0 ? "+" : "") + n.toLocaleString('pt-BR', {minimumFractionDigits: 1, maximumFractionDigits: 1});
  return s + "%";
}
""")

        row_style = make_row_style_js()
        on_ready, on_col_vis = on_fit_events_js()

        def col_num(field, header=None, formatter=None):
            return {
                "headerName": header or field,
                "field": field,
                "editable": False,
                "type": "numericColumn",
                "valueFormatter": formatter or fmt,
                "cellClass": "ag-right-aligned-cell",
            }

        df5_view = df5[['indicador_id', 'Indicador', 'BP25 (FY)', 'FY (YTD+YTG)', 'Δ Abs', 'Δ %']].rename(
            columns={'Indicador': 'Linha P&L'}).copy()

        column_defs = [
            {"field": "indicador_id", "hide": True},
            {"headerName": "Linha P&L", "field": "Linha P&L", "pinned": "left", "editable": False},
            col_num('BP25 (FY)'),
            col_num('FY (YTD+YTG)'),
            col_num('Δ Abs'),
            col_num('Δ %', formatter=fmt_pct),
        ]

        grid_options = {
            "columnDefs": column_defs,
            "defaultColDef": {"resizable": True, "sortable": False, "filter": False,
                              "cellStyle": {"fontVariantNumeric": "tabular-nums"}},
            "getRowStyle": row_style,
            "rowHeight": 34, "suppressMovableColumns": True, "ensureDomOrder": True,
            "onFirstDataRendered": on_ready, "onColumnVisible": on_col_vis,
            "domLayout": "autoHeight",  # <- sem rolagem
        }

        AgGrid(
            df5_view,
            gridOptions=grid_options,
            data_return_mode="AS_INPUT",
            update_mode=GridUpdateMode.NO_UPDATE,
            allow_unsafe_jscode=True,
            theme="balham",
            fit_columns_on_grid_load=True,
            key=f"bp_vs_fy_grid_{year}_{scale}_{int(st.session_state.get('show_all_rows_t2', False))}",
        )

    # ==================================================================================
    # ABA 3 — RES vs UI | 5 colunas
    # ==================================================================================
    with tab3:
        st.markdown("#### RES Working vs Projeção UI")

        colf1, _ = st.columns([2, 1])
        with colf1:
            st.checkbox("Detalhado", key="show_all_rows_t3", help="Exibir todas as linhas.")

        # RES_WORKING pivot (YTD realizado + YTG res_working)
        res_piv = build_res_working_pivot(df, year, cutoff)

        # UI pivot (YTD realizado + YTG UI)
        if piv_real is not None:
            ui_piv = build_simulado_pivot(
                df=df,
                piv_real=piv_real,
                year=year,
                cutoff=cutoff,
                base_calc_path=BASE_CALC_PATH,
                volumes_edit=volumes_ui_long,  # Usa UI
                volumes_res=pd.DataFrame(),  # Não usa RES
                volume_mode="ui",
                dme_pct=0.102,
                ui_month_totals=ui_totals,
                conv_source="excel",
            )
        else:
            # Fallback se não houver dados realizados
            ui_piv = pivot_pnl_alias(df, year, baseline_name or scens[0])

        req_cols = ['indicador_id', 'Indicador', 'Total Ano']
        res_view = res_piv[req_cols].rename(columns={'Total Ano': 'RES Working'}).copy()
        ui_view = ui_piv[req_cols].rename(columns={'Total Ano': 'UI Projeção'}).copy()

        # Combina os dataframes
        order_ids = list(res_view['indicador_id'])
        df_res_ui = res_view.merge(ui_view[['indicador_id', 'UI Projeção']], on='indicador_id', how='left')
        extras = ui_view[~ui_view['indicador_id'].isin(df_res_ui['indicador_id'])]
        if not extras.empty:
            df_res_ui = pd.concat([df_res_ui, extras], ignore_index=True, sort=False)
            for x in extras['indicador_id'].tolist():
                if x not in order_ids:
                    order_ids.append(x)

        df_res_ui['__ord__'] = pd.Categorical(df_res_ui['indicador_id'], categories=order_ids, ordered=True)
        df_res_ui = df_res_ui.sort_values('__ord__').drop(columns='__ord__')

        # Filtro para linhas detalhadas
        if not st.session_state.get("show_all_rows_t3", False):
            keep_ids = set(BOLD_GREY_ROWS)
            df_res_ui = df_res_ui[df_res_ui['indicador_id'].isin(keep_ids)].reset_index(drop=True)

        # Cálculos de diferença
        df_res_ui['RES Working'] = df_res_ui['RES Working'].fillna(0.0)
        df_res_ui['UI Projeção'] = df_res_ui['UI Projeção'].fillna(0.0)
        df_res_ui['Δ Abs'] = df_res_ui['UI Projeção'] - df_res_ui['RES Working']
        res_safe = df_res_ui['RES Working'].replace({0: np.nan})
        df_res_ui['Δ %'] = ((df_res_ui['UI Projeção'] - df_res_ui['RES Working']) / res_safe) * 100.0

        # -------- Tabela estilo AgGrid (sem rolagem) --------
        escala_label = st.session_state.get(S._SCALE_LABEL_KEY, "1.000x")
        scale = {"1x": 1, "1.000x": 1_000, "1.000.000x": 1_000_000}[escala_label]
        fmt = make_value_formatter(scale)

        fmt_pct = JsCode("""
function(params){
  var v = params.value;
  if (v === null || v === undefined) return "-";
  var n = Number(v);
  if (isNaN(n)) return "-";
  var s = (n >= 0 ? "+" : "") + n.toLocaleString('pt-BR', {minimumFractionDigits: 1, maximumFractionDigits: 1});
  return s + "%";
}
""")

        row_style = make_row_style_js()
        on_ready, on_col_vis = on_fit_events_js()

        def col_num(field, header=None, formatter=None):
            return {
                "headerName": header or field,
                "field": field,
                "editable": False,
                "type": "numericColumn",
                "valueFormatter": formatter or fmt,
                "cellClass": "ag-right-aligned-cell",
            }

        df_res_ui_view = df_res_ui[['indicador_id', 'Indicador', 'RES Working', 'UI Projeção', 'Δ Abs', 'Δ %']].rename(
            columns={'Indicador': 'Linha P&L'}).copy()

        column_defs = [
            {"field": "indicador_id", "hide": True},
            {"headerName": "Linha P&L", "field": "Linha P&L", "pinned": "left", "editable": False},
            col_num('RES Working'),
            col_num('UI Projeção'),
            col_num('Δ Abs'),
            col_num('Δ %', formatter=fmt_pct),
        ]

        grid_options = {
            "columnDefs": column_defs,
            "defaultColDef": {"resizable": True, "sortable": False, "filter": False,
                              "cellStyle": {"fontVariantNumeric": "tabular-nums"}},
            "getRowStyle": row_style,
            "rowHeight": 34, "suppressMovableColumns": True, "ensureDomOrder": True,
            "onFirstDataRendered": on_ready, "onColumnVisible": on_col_vis,
            "domLayout": "autoHeight",  # <- sem rolagem
        }

        AgGrid(
            df_res_ui_view,
            gridOptions=grid_options,
            data_return_mode="AS_INPUT",
            update_mode=GridUpdateMode.NO_UPDATE,
            allow_unsafe_jscode=True,
            theme="balham",
            fit_columns_on_grid_load=True,
            key=f"res_vs_ui_grid_{year}_{scale}_{int(st.session_state.get('show_all_rows_t3', False))}",
        )

        # Adiciona informações explicativas
        with st.expander("ℹ️ Sobre esta comparação"):
            st.markdown("""
            **RES Working**: Combina YTD (dados realizados) + YTG (projeção do arquivo RES_WORKING)

            **UI Projeção**: Combina YTD (dados realizados) + YTG (projeção inserida manualmente na interface)

            **Δ Abs**: Diferença absoluta (UI Projeção - RES Working)

            **Δ %**: Diferença percentual ((UI Projeção - RES Working) / RES Working * 100)
            """)

    # -------------------------
    # Diagnóstico no rodapé
    # -------------------------
    st.divider()
    st.markdown("#### 📁 Diagnóstico de Arquivos")
    debug_parquet_status()


if __name__ == "__main__":
    main()