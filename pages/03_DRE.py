import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
from core.state import init_state
from core.io import load_volume_base

st.set_page_config(page_title="P&L (Template)", page_icon="📈", layout="wide")
init_state()

st.header("📈 P&L")

# =========================
# Linhas FIXAS do modelo
# =========================
LINHAS_FIXAS = [
    "Volume Sell Out (UC)",
    "Volume (UC)",
    "Receita Bruta",
    "Convênio/Impostos",
    "Receita Líquida",
    "Insumos",
    "Custos Toll Packer",
    "Concentrado/Incentivo de Vendas",
    "Fretes T1",
    "Estrutura Industrial - Variável",
    "Custos Variáveis",
    "Margem Variável",
    "Estrutura Industrial - Fixa Reg",
    "Margem Bruta",
    "Custos Log. T1 Reg",
    "Despesas Secundárias (T2)",
    "Perdas",
    "Margem Contribuição",
    "DME",
    "Margem Contribuição Líquida",
    "Ajuste Cota Fixa",
    "Opex Leão Reg",
    "Cluster CCIL",
    "Charge Back",
    "R. Oper. antes Regionalização",
    "Custo Regionalizacao",
    "Resultado Operacional",
    "Depreciacao",
    "EBITDA",
]

# Seções com destaque (cinza + negrito)
SECOES_NEGRITO = {
    "Custos Variáveis", "Margem Variável", "Margem Bruta",
    "Margem Contribuição", "Margem Contribuição Líquida",
    "R. Oper. antes Regionalização", "Resultado Operacional", "EBITDA"
}

# =========================
# Auxiliares
# =========================
MAP_M = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}
MESES = list(range(1,13))

def fmt_br_int(x, factor=1):
    """Formata número com separador de milhar pt-BR. Se factor != 1, mostra com 2 casas."""
    try:
        val = float(x) / float(factor)
        if factor == 1:
            return f"{int(round(val)):,}".replace(",", ".")
        else:
            return f"{val:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
    except Exception:
        return x

def fmt_pct(v):
    try:
        return f"{float(v)*100:,.1f}%".replace(",", "_").replace(".", ",").replace("_", ".")
    except Exception:
        return "—"

def monthly_volume_from_base(df_vol_base: pd.DataFrame, ano_sel: int) -> pd.Series:
    df = df_vol_base.copy()
    df = df[pd.to_numeric(df["ano"], errors="coerce") == ano_sel]
    if df.empty:
        return pd.Series({m: 0 for m in MESES}, dtype=int)
    grp = df.groupby("mes")["volume"].sum().reindex(MESES, fill_value=0)
    return pd.to_numeric(grp, errors="coerce").fillna(0).round(0).astype(int)

def monthly_volume_from_sim(volumes_edit: pd.DataFrame|None, ano_sel: int) -> pd.Series:
    if volumes_edit is None:
        return pd.Series({m: 0 for m in MESES}, dtype=int)
    df = volumes_edit.copy()
    df = df[pd.to_numeric(df["ano"], errors="coerce") == ano_sel]
    if df.empty:
        return pd.Series({m: 0 for m in MESES}, dtype=int)
    grp = df.groupby("mes")["volume"].sum().reindex(MESES, fill_value=0)
    return pd.to_numeric(grp, errors="coerce").fillna(0).round(0).astype(int)

def build_month_cols(meses):
    return [MAP_M[m] for m in meses]

def build_comp_cols(meses):
    cols = []
    for m in meses:
        cols += [f"{MAP_M[m]} Base", f"{MAP_M[m]} Sim", f"{MAP_M[m]} Δ", f"{MAP_M[m]} Δ%"]
    return cols

def make_format_dict(df, factor):
    def make_formatter(col):
        if col.endswith("Δ%") or col == "Δ%":
            return lambda v: fmt_pct(v) if (v is not None) else "—"
        else:
            return lambda v: fmt_br_int(v, factor)

    fmt = {}
    for col in df.columns:
        if col == "Indicador":
            continue
        fmt[col] = make_formatter(col)
    return fmt

# =========================
# Base de volumes
# =========================
vol_path = Path("data/base_final_longo_para_modelo.xlsx")
if not vol_path.exists():
    st.warning("Base de volumes não encontrada em `data/base_final_longo_para_modelo.xlsx`. Abra a página **Volumes** para fazer upload.")
    st.stop()

df_vol_base = load_volume_base(vol_path)
df_vol_base["ano"] = pd.to_numeric(df_vol_base["ano"], errors="coerce")
anos = sorted([int(a) for a in df_vol_base["ano"].dropna().unique().tolist()])
if not anos:
    st.error("Não foi possível identificar anos válidos na base de volumes.")
    st.stop()

# =========================
# Seletores
# =========================
c1, c2, c3 = st.columns([1,1.4,1.6])
with c1:
    ano_sel = st.selectbox("Ano", options=anos, index=len(anos)-1)
with c2:
    modo = st.selectbox("Cenário", options=["Full Year", "Simulado", "Comparativo"])
with c3:
    escala_lbl = st.selectbox("Escala", options=["1x","1.000x","1.000.000x"], index=0)
scale_factor = {"1x":1, "1.000x":1_000, "1.000.000x":1_000_000}[escala_lbl]

volumes_edit = st.session_state.get("volumes_edit")

# Volumes mensais (Base e Sim)
base_m = monthly_volume_from_base(df_vol_base, ano_sel)
# depois — garante fallback para FY quando não houver simulação carregada
sim_raw = monthly_volume_from_sim(volumes_edit, ano_sel)

# Se não existe volumes_edit (ou ficou zerado), usa FY (base)
if (volumes_edit is None) or (sim_raw.sum() == 0):
    sim_m = base_m.copy()
else:
    # Caso raro: se algum mês vier ausente/NaN na simulação, preenche com FY
    sim_m = sim_raw.combine_first(base_m).astype(int)

delta_m = (sim_m - base_m).astype(int)

# =========================
# Builders das tabelas
# =========================
def make_table_full_year(series_by_line: dict):
    """Gera tabela Jan–Dez + Total."""
    cols = build_month_cols(MESES) + ["Total"]
    rows = []
    for nome in LINHAS_FIXAS:
        r = {"Indicador": nome}
        if nome in series_by_line:
            s = series_by_line[nome]
            for m in MESES:
                r[MAP_M[m]] = int(s.loc[m])
            r["Total"] = int(s.sum())
        else:
            for c in cols:
                r[c] = 0
        rows.append(r)
    df = pd.DataFrame(rows, columns=["Indicador"] + cols)
    return df

def make_table_comparativo_mensal(only_changed: bool):
    meses_view = [m for m in MESES if delta_m.loc[m] != 0] if only_changed else MESES
    if not meses_view:
        meses_view = MESES
    cols = build_comp_cols(meses_view)
    rows = []
    for nome in LINHAS_FIXAS:
        r = {"Indicador": nome}
        if nome in ("Volume (UC)", "Volume Sell Out (UC)"):
            for m in meses_view:
                base_v = int(base_m.loc[m])
                sim_v  = int(sim_m.loc[m])
                d      = sim_v - base_v
                pct    = (d / base_v) if base_v != 0 else None
                r[f"{MAP_M[m]} Base"] = base_v
                r[f"{MAP_M[m]} Sim"]  = sim_v
                r[f"{MAP_M[m]} Δ"]    = d
                r[f"{MAP_M[m]} Δ%"]   = pct
        else:
            for c in cols:
                r[c] = 0
        rows.append(r)
    return pd.DataFrame(rows, columns=["Indicador"] + cols)

def make_table_comparativo_ytd_ytg(cutoff_month: int, show_ytd_detail: bool):
    """Retorna dois DataFrames: df_ytd (agregado, com opção de detalhar meses YTD) e df_ytg (mensal para meses > cutoff)."""
    ytd_months = [m for m in MESES if m <= cutoff_month]
    ytg_months = [m for m in MESES if m > cutoff_month]

    # ----- YTD agregado -----
    rows_ytd = []
    for nome in LINHAS_FIXAS:
        r = {"Indicador": nome}
        if nome in ("Volume (UC)", "Volume Sell Out (UC)"):
            b = int(base_m.loc[ytd_months].sum())
            s = int(sim_m.loc[ytd_months].sum())
            d = s - b
            p = (d / b) if b != 0 else None
            r.update({"Base": b, "Sim": s, "Δ": d, "Δ%": p})
        else:
            r.update({"Base": 0, "Sim": 0, "Δ": 0, "Δ%": None})
        rows_ytd.append(r)
    df_ytd = pd.DataFrame(rows_ytd, columns=["Indicador","Base","Sim","Δ","Δ%"])

    # (opcional) detalhar YTD por mês
    if show_ytd_detail and ytd_months:
        cols_det = build_comp_cols(ytd_months)
        for i, nome in enumerate(LINHAS_FIXAS):
            if nome in ("Volume (UC)", "Volume Sell Out (UC)"):
                detail = {}
                for m in ytd_months:
                    b = int(base_m.loc[m]); s = int(sim_m.loc[m]); d = s - b; p = (d/b) if b else None
                    detail[f"{MAP_M[m]} Base"] = b
                    detail[f"{MAP_M[m]} Sim"]  = s
                    detail[f"{MAP_M[m]} Δ"]    = d
                    detail[f"{MAP_M[m]} Δ%"]   = p
                df_ytd.loc[i, cols_det] = pd.Series(detail)

    # ----- YTG aberto (mensal) -----
    rows_ytg = []
    cols_ytg = build_comp_cols(ytg_months) if ytg_months else []
    for nome in LINHAS_FIXAS:
        r = {"Indicador": nome}
        if nome in ("Volume (UC)", "Volume Sell Out (UC)"):
            for m in ytg_months:
                b = int(base_m.loc[m]); s = int(sim_m.loc[m]); d = s - b; p = (d/b) if b else None
                r[f"{MAP_M[m]} Base"] = b
                r[f"{MAP_M[m]} Sim"]  = s
                r[f"{MAP_M[m]} Δ"]    = d
                r[f"{MAP_M[m]} Δ%"]   = p
        else:
            for c in cols_ytg:
                r[c] = 0
        rows_ytg.append(r)
    df_ytg = pd.DataFrame(rows_ytg, columns=["Indicador"] + cols_ytg)

    return df_ytd, df_ytg

# =========================
# Estilos (DataFrame Styler)
# =========================
def style_sections(row, df_ref):
    styles = [''] * len(row)
    # Seções em cinza/negrito
    if row["Indicador"] in SECOES_NEGRITO:
        for i in range(len(styles)):
            styles[i] = 'font-weight:700; background-color:#f3f4f6;'
    # Linha inteira de Volume Sell Out (UC) em negrito
    if row["Indicador"] == "Volume Sell Out (UC)":
        for i in range(len(styles)):
            styles[i] = (styles[i] + ' font-weight:800;') if styles[i] else 'font-weight:800;'
    return styles

def style_deltas(row, df_ref):
    styles = [''] * len(row)
    # Destaca Δ e Δ% quando ≠ 0
    for i, col in enumerate(df_ref.columns):
        if col.endswith("Δ") and isinstance(row[col], (int, float)) and row[col] != 0:
            styles[i] = 'background-color:#fde68a; font-weight:600;'
        if col.endswith("Δ%") and (row[col] is not None) and row[col] != 0:
            styles[i] = 'background-color:#fde68a; font-weight:600;'
    return styles

# =========================
# Render por modo
# =========================
if modo == "Full Year":
    series_map = {
        "Volume (UC)": base_m,
        "Volume Sell Out (UC)": base_m,  # ajuste se quiser diferenciar
    }
    df_view = make_table_full_year(series_map)
    fmt_dict = make_format_dict(df_view, scale_factor)
    st.metric("Total Ano (Full Year)", fmt_br_int(int(base_m.sum()), scale_factor))
    st.dataframe(
        df_view.style.apply(lambda r: style_sections(r, df_view), axis=1).format(fmt_dict),
        use_container_width=True, height=560
    )

elif modo == "Simulado":
    series_map = {
        "Volume (UC)": sim_m,
        "Volume Sell Out (UC)": sim_m,
    }
    df_view = make_table_full_year(series_map)
    fmt_dict = make_format_dict(df_view, scale_factor)
    st.metric("Total Ano (Simulado)", fmt_br_int(int(sim_m.sum()), scale_factor))
    st.dataframe(
        df_view.style.apply(lambda r: style_sections(r, df_view), axis=1).format(fmt_dict),
        use_container_width=True, height=560
    )

else:  # Comparativo
    base_fy = int(base_m.sum())
    sim_fy  = int(sim_m.sum())
    delta_fy = sim_fy - base_fy
    c_y1, c_y2, c_y3 = st.columns(3)
    c_y1.metric("Base (FY)", fmt_br_int(base_fy, scale_factor))
    c_y2.metric("Sim (FY)", fmt_br_int(sim_fy, scale_factor))
    c_y3.metric("Δ (FY)", fmt_br_int(delta_fy, scale_factor),
                delta=fmt_br_int(delta_fy, scale_factor))

    tab_mensal, tab_ytd = st.tabs(["Mensal", "YTD / YTG"])

    with tab_mensal:
        only_changed = st.checkbox("Mostrar apenas meses com alteração", value=False, key="cmp_only_changed")
        df_mensal = make_table_comparativo_mensal(only_changed)
        fmt_month = make_format_dict(df_mensal, scale_factor)
        st.dataframe(
            df_mensal.style
                .apply(lambda r: style_sections(r, df_mensal), axis=1)
                .apply(lambda r: style_deltas(r, df_mensal), axis=1)
                .format(fmt_month),
            use_container_width=True, height=560
        )

    with tab_ytd:
        # Mês de corte (YTD)
        curr_month = datetime.now().month
        cutoff = st.selectbox(
            "Mês de corte (YTD)",
            options=[MAP_M[m] for m in MESES],
            index=(curr_month - 1) if 1 <= curr_month <= 12 else 6
        )
        cutoff_month = [m for m in MESES if MAP_M[m] == cutoff][0]
        show_ytd_detail = st.checkbox("Mostrar meses YTD (detalhe)", value=False, key="ytd_detail")

        df_ytd, df_ytg = make_table_comparativo_ytd_ytg(cutoff_month, show_ytd_detail)

        # YTD agregado (com Δ%)
        fmt_ytd = make_format_dict(df_ytd, scale_factor)
        st.markdown("**YTD (acumulado até o mês de corte)**")
        st.dataframe(
            df_ytd.style
                .apply(lambda r: style_sections(r, df_ytd), axis=1)
                .apply(lambda r: style_deltas(r, df_ytd), axis=1)
                .format(fmt_ytd),
            use_container_width=True
        )

        # YTG aberto
        st.markdown("**YTG (meses restantes do ano)**")
        if df_ytg.shape[1] <= 1:
            st.info("Não há meses restantes após o mês de corte selecionado.")
        else:
            fmt_ytg = make_format_dict(df_ytg, scale_factor)
            st.dataframe(
                df_ytg.style
                    .apply(lambda r: style_sections(r, df_ytg), axis=1)
                    .apply(lambda r: style_deltas(r, df_ytg), axis=1)
                    .format(fmt_ytg),
                use_container_width=True
            )

st.caption(
    "Cenários: **Full Year** (Jan–Dez + Total), **Simulado** (Jan–Dez + Total), "
    "**Comparativo** (Mensal com Base|Sim|Δ|Δ% e YTD agregado com opção de detalhar + YTG aberto). "
    "Somente as linhas de **Volume** são populadas; as demais linhas serão calculadas depois."
)
