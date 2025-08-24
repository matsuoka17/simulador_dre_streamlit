# pages/03_DRE.py
# --------------------------------------------------------------------------------------
# DRE (Parquet) — 2 abas no MESMO arquivo:
#   1) P&L Mensal (a visão que você já usa — intacta)
#   2) BP25 vs FY (YTD+YTG) — 5 colunas (Linha, BP, FY, Δ Abs, Δ %)
# --------------------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from io import BytesIO
import re

import numpy as np
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.shared import JsCode

# Layout sempre largo
st.set_page_config(page_title="DRE", page_icon="📊", layout="wide")

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
from core.calculator import build_simulado_pivot
import core.sim as S  # chaves de sessão globais (toggle/escala)

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

MONTHS_LOWER = {name.lower(): (m, name) for m, name in MONTHS_PT.items()}

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
    col_map = {}
    for c in dfw.columns:
        lc = str(c).strip().lower()
        if lc in MONTHS_LOWER:
            col_map[MONTHS_LOWER[lc][0]] = c
    totals = {}
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
    # usa models.res_volume_total_by_month (RES_WORKING)
    d = res_volume_total_by_month(year)
    if not d:
        return None
    return {int(k): int(v) for k, v in d.items()}

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
            return {{'backgroundColor':'#F3F4F6','fontWeight':'600'}};
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
        # Toggle global de projeção (UI → YTG)
        use_sim = st.toggle("Usar Projeção (UI) para YTG", value=bool(st.session_state.get(S._USE_SIM_KEY, True)))
        st.session_state[S._USE_SIM_KEY] = use_sim

        escala_label = st.selectbox(
            "Escala",
            ["1x", "1.000x", "1.000.000x"],
            index=["1x","1.000x","1.000.000x"].index(st.session_state.get(S._SCALE_LABEL_KEY, "1x"))
        )
        st.session_state[S._SCALE_LABEL_KEY] = escala_label
        # espelha na chave antiga usada em captions/export (sem criar seletor no topo)
        st.session_state["dre_scale_label"] = escala_label

        st.caption("• YTD vem do cenário Realizado do CURRENT.\n• YTG usa UI se marcado; caso contrário, RES_WORKING.")

        st.divider()
        st.header("📁 Status dos Arquivos")
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

    # Estado inicial (sem seletor de escala no topo!)
    st.session_state.setdefault("dre_year", default_year)

    st.markdown("### Demonstração de Resultado (DRE) — Comparativo P&L Mensal")

    # Controles COMUNS às abas (apenas calcula escala/ano)
    year = st.session_state.get("dre_year") or default_year
    escala_label = st.session_state.get(S._SCALE_LABEL_KEY, "1x")
    scale = {"1x": 1, "1.000x": 1_000, "1.000.000x": 1_000_000}[escala_label]

    # Cenários
    scens = scenarios_for_year(df, year)
    if not scens:
        st.warning("Nenhum cenário para o ano selecionado.")
        st.stop()

    scen_real = scenario_realizado_for_year(df, year)
    cutoff = cutoff_from_realizado(df, year)
    baseline_name = find_baseline_scenario(df, year)

    # Pré-cálculos: volumes UI/RES
    volumes_ui_long = st.session_state.get("volumes_edit")
    volumes_res_df = res_volume_by_family_long()  # baseline YTG sempre RES_WORKING
    ui_totals = get_volumes_from_ui_total(year) if st.session_state.get(S._USE_SIM_KEY, True) else None
    if ui_totals is None:
        # fallback para RES quando UI não existir/estiver vazia
        ui_totals = get_volumes_from_res(year)

    piv_real = pivot_pnl_alias(df, year, scen_real) if scen_real else None
    piv_bp = pivot_pnl_alias(df, year, baseline_name) if baseline_name else None

    # ----------------- ABAS -----------------
    tab1, tab2 = st.tabs(["P&L Mensal", "BP25 vs Projeção"])

    # ==================================================================================
    # ABA 1 — P&L MENSAL
    # ==================================================================================
    with tab1:
        # Cenário para comparativo (KPIs)
        scen_sel = st.selectbox(
            "Cenário para comparativo",
            scens,
            index=min(1, len(scens)-1) if len(scens) > 1 else 0,
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

        # Projeção (simulado) — usa APENAS o toggle global da lateral
        use_sim = bool(st.session_state.get(S._USE_SIM_KEY, True))
        if piv_real is not None:
            piv_sim = build_simulado_pivot(
                df=df,
                piv_real=piv_real,
                year=year,
                cutoff=cutoff,
                base_calc_path=BASE_CALC_PATH,
                volumes_edit=(volumes_ui_long if use_sim else None),
                volumes_res=volumes_res_df,
                volume_mode=("ui" if use_sim else "res"),
                dme_pct=0.102,
                ui_month_totals=ui_totals,
                conv_source="excel",
            )
        else:
            piv_sim = None

        with c1:
            st.markdown('<div class="sim-badge">Simulado</div>', unsafe_allow_html=True)
            kpi_card("Volume (UC)",
                     sum_row(piv_sim, "volume_uc") if piv_sim is not None else 0,
                     sum_row(piv_sel, "volume_uc"))
        with c2:
            kpi_card("Receita Líquida",
                     sum_row(piv_sim, "receita_liquida") if piv_sim is not None else 0,
                     sum_row(piv_sel, "receita_liquida"))
        with c3:
            kpi_card("Resultado Operacional",
                     sum_row(piv_sim, "resultado_operacional") if piv_sim is not None else 0,
                     sum_row(piv_sel, "resultado_operacional"))

        st.markdown("""
        <style>
        .sim-badge{background:#0ea5e9;color:white;padding:4px 8px;border-radius:8px;
                   font-size:12px;text-align:center;width:80px}
        .ag-right-aligned-cell{ text-align:right !important; font-variant-numeric: tabular-nums; }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Controles da tabela (sem toggle de projeção aqui!)
        ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.2, 1.0, 1.0, 1.6], gap="small")
        with ctrl1:
            st.checkbox("Detalhado", key="show_all_rows_t1", help="Exibir todas as linhas.")
        with ctrl2:
            st.checkbox("YTD", key="show_ytd_detail_t1")
        with ctrl3:
            st.checkbox("YTG", key="show_ytg_detail_t1")

        # Cenário Base da Tabela
        default_table_scen = scenario_realizado_for_year(df, year) or scens[0]
        st.selectbox(
            "Cenário Base da Tabela",
            scens,
            index=scens.index(default_table_scen) if default_table_scen in scens else 0,
            key="dre_table_scen"
        )

        piv_table = pivot_pnl_alias(df, year, st.session_state["dre_table_scen"])

        # Aplica simulação quando base = Realizado (usa só o toggle da lateral)
        if is_realizado(st.session_state["dre_table_scen"]) and piv_real is not None:
            piv_table = build_simulado_pivot(
                df=df, piv_real=piv_table, year=year, cutoff=cutoff,
                base_calc_path=BASE_CALC_PATH,
                volumes_edit=(volumes_ui_long if use_sim else None),
                volumes_res=volumes_res_df,
                volume_mode=("ui" if use_sim else "res"),
                dme_pct=0.102,
                ui_month_totals=(get_volumes_from_ui_total(year) if use_sim else get_volumes_from_res(year)),
                conv_source="excel",
            )

        # Export Excel (tabela mensal completa)
        with ctrl4:
            scen_slug = re.sub(r'[^A-Za-z0-9]+', '-', str(_short_title(st.session_state["dre_table_scen"])))
            tag = "Proj" if (is_realizado(st.session_state["dre_table_scen"]) and use_sim) else "Parquet"
            st.download_button(
                "Exportar Excel (P&L mensal)",
                data=make_excel_bytes(piv_table, sheet_name="P&L Mensal"),
                file_name=f"DRE_{year}_{scen_slug}_{tag}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        # Visão (todas x principais)
        if st.session_state["show_all_rows_t1"]:
            piv_table_view = piv_table.copy()
        else:
            keep_ids = set(BOLD_GREY_ROWS)
            piv_table_view = piv_table[piv_table["indicador_id"].isin(keep_ids)].reset_index(drop=True)

        # Colunas agrupadas YTD / YTG
        ytd_cols = [v for k, v in MONTHS_PT.items() if k <= cutoff]
        ytg_cols = [v for k, v in MONTHS_PT.items() if k > cutoff]

        fmt = make_value_formatter(scale)
        row_style = make_row_style_js()
        on_ready, on_col_vis = on_fit_events_js()

        def child(col_name: str, hidden: bool = False):
            return {
                "field": col_name,
                "editable": False,
                "type": "numericColumn",
                "valueFormatter": fmt,
                "hide": hidden,
                "cellClass": "ag-right-aligned-cell",
            }

        ytd_children = [child(c, hidden=(not st.session_state["show_ytd_detail_t1"])) for c in ytd_cols]
        ytg_children = [child(c, hidden=(not st.session_state["show_ytg_detail_t1"])) for c in ytg_cols]

        column_defs = [
            {"field": "indicador_id", "hide": True},
            {"headerName": "Indicador", "field": "Indicador", "pinned": "left", "editable": False},
        ]
        if ytd_children:
            column_defs.append({
                "headerName": f"YTD (<= M{cutoff:02d})" if cutoff > 0 else "YTD",
                "marryChildren": True, "children": ytd_children,
            })
        if ytg_children:
            column_defs.append({
                "headerName": f"YTG (>{cutoff:02d})" if cutoff > 0 else "YTG",
                "marryChildren": True, "children": ytg_children,
            })

        existing = {c["field"] for c in ytd_children + ytg_children}
        for m in range(1, 13):
            mc = MONTHS_PT[m]
            if mc not in existing:
                column_defs.append(child(mc, hidden=True))
        column_defs.append(child("Total Ano", hidden=False))

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
        grid_key = f"dre_grid_{year}_{scen_slug}_{int(st.session_state['show_ytd_detail_t1'])}_{int(use_sim)}_{int(st.session_state['show_all_rows_t1'])}_{int(st.session_state['show_ytg_detail_t1'])}_{scale}"

        AgGrid(
            piv_table_view,
            gridOptions=grid_options,
            data_return_mode="AS_INPUT",
            update_mode=GridUpdateMode.NO_UPDATE,
            allow_unsafe_jscode=True,
            theme="balham",
            height=560,
            fit_columns_on_grid_load=True,
            key=grid_key,
        )

    # ==================================================================================
    # ABA 2 — BP25 vs FY (YTD+YTG) | 5 colunas
    # ==================================================================================
    with tab2:
        st.markdown("#### BP25 vs Projeção")

        # Controles (sem toggle aqui também)
        colf1, _ = st.columns([2, 1])
        with colf1:
            st.checkbox(
                "Detalhado",
                key="show_all_rows_t2",
                help="Exibir todas as linhas."
            )

        bp_piv = piv_bp if piv_bp is not None else pivot_pnl_alias(df, year, baseline_name or scens[0])

        use_sim2 = bool(st.session_state.get(S._USE_SIM_KEY, True))
        if piv_real is not None:
            fy_piv = build_simulado_pivot(
                df=df, piv_real=piv_real, year=year, cutoff=cutoff,
                base_calc_path=BASE_CALC_PATH,
                volumes_edit=(volumes_ui_long if use_sim2 else None),
                volumes_res=volumes_res_df,
                volume_mode=("ui" if use_sim2 else "res"),
                dme_pct=0.102,
                ui_month_totals=(get_volumes_from_ui_total(year) if use_sim2 else get_volumes_from_res(year)),
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

        # formatação com a ESCALA da barra lateral
        escala_label = st.session_state.get(S._SCALE_LABEL_KEY, "1x")
        scale = {"1x": 1, "1.000x": 1_000, "1.000.000x": 1_000_000}[escala_label]

        def _fmt_int(v):
            if pd.isna(v): return '-'
            try:
                return f"{int(np.rint(v / scale)):,}".replace(',', '.')
            except Exception:
                return '-'

        def _fmt_pct(v):
            if pd.isna(v): return '-'
            return f"{v:+.1f}%"

        out5 = df5[['Indicador', 'BP25 (FY)', 'FY (YTD+YTG)', 'Δ Abs', 'Δ %']].rename(columns={'Indicador': 'Linha P&L'})

        yellow_rows = df5['indicador_id'].eq('resultado_operacional').reindex(out5.index, fill_value=False).tolist()

        def _row_style(row):
            i = row.name
            if yellow_rows[i]:
                return ['background-color: #FFF8C5; font-weight:700'] * len(row)
            if df5['indicador_id'].iloc[i] in BOLD_GREY_ROWS:
                return ['background-color: #F3F4F6; font-weight:600'] * len(row)
            return [''] * len(row)

        st.dataframe(
            out5.style
                .format({
                    'BP25 (FY)': _fmt_int,
                    'FY (YTD+YTG)': _fmt_int,
                    'Δ Abs': _fmt_int,
                    'Δ %': _fmt_pct,
                })
                .apply(_row_style, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(f"Valores na escala **{escala_label}**. Δ% calculado sobre o BP (divide por zero exibido como “-”).")

if __name__ == "__main__":
    main()
