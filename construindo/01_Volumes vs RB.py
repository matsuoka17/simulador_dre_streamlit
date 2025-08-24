# pages/01_Volume_YTG_Mensal.py
# -----------------------------------------------------------------------------
# Volume YTG mensal por Família (absoluto) + Preço Dez (R$/UC) da linha "Receita Bruta" (Excel)
# + Mesma tabela para: Insumos, Custos Toll Packer, Fretes T1
#
# Fontes:
#   - Volumes (RES parquet): core.models.re_volume_by_family_long()
#   - Cutoff (RE):           core.models.realized_cutoff_by_year()
#   - Parâmetros (Excel):    data/premissas_pnl/base_calculos.xlsx (Indicador; Valor por mês)
#
# Saída principal (RB):
#   Família | Preço Dez (R$/UC) | [Meses YTG em colunas] + [RB mês = Volume*Preço] |
#   Total YTG | Total Volume YTG (UC) | Total RB YTG (R$)
#
# Saídas adicionais (Insumos/Toll/Frete):
#   Família | Param Dez (R$/UC) | [Meses YTG] + [Linha mês (R$) = Volume*Param] |
#   Total YTG | Total Volume YTG (UC) | Total <Linha> YTG (R$)
#
# Além disso, expõe totais mensais globais (soma de famílias) via variáveis globais e session_state:
#   - RB:     rb_ytg_total_by_month,   rb_ytg_total_fy
#   - Insumos:insumos_ytg_total_by_month, insumos_ytg_total_fy
#   - Toll:   toll_ytg_total_by_month, toll_ytg_total_fy
#   - Frete:  frete_ytg_total_by_month, frete_ytg_total_fy
# -----------------------------------------------------------------------------

from __future__ import annotations
import io
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Volume YTG Mensal + Preço Dez", page_icon="🗓️", layout="wide")

# Base do projeto (volumes/cutoff)
from core.models import re_volume_by_family_long, realized_cutoff_by_year

MONTHS_PT = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}
BASE_CALCULOS_XLSX = Path("data/premissas_pnl/base_calculos.xlsx")
MESES_MAP_NUM = {"jan":1,"fev":2,"mar":3,"abr":4,"mai":5,"jun":6,"jul":7,"ago":8,"set":9,"out":10,"nov":11,"dez":12}

# ============================== EXPORTS GLOBAIS ============================= #
# Totais para reutilização por outras páginas (ex.: DRE), sem recálculo:
VOL_YTG_TOTAL_BY_MONTH: dict[str, float] = {}
RB_YTG_TOTAL_BY_MONTH: dict[str, float] = {}
VOL_YTG_TOTAL_FY: float = 0.0
RB_YTG_TOTAL_FY: float = 0.0

INSUMOS_YTG_TOTAL_BY_MONTH: dict[str, float] = {}
TOLL_YTG_TOTAL_BY_MONTH: dict[str, float] = {}
FRETE_YTG_TOTAL_BY_MONTH: dict[str, float] = {}
INSUMOS_YTG_TOTAL_FY: float = 0.0
TOLL_YTG_TOTAL_FY: float = 0.0
FRETE_YTG_TOTAL_FY: float = 0.0

YTG_MONTHS: list[str] = []
YTG_ANO: int | None = None
YTG_CUTOFF: int | None = None
# =========================================================================== #

# -------------------------------- Utils ------------------------------------- #
def _fmt_int_dot(x) -> str:
    try:
        v = int(round(float(x)))
    except Exception:
        v = 0
    return f"{v:,}".replace(",", ".")

def _fmt_6dec_ptbr(x) -> str:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    s = f"{v:,.6f}"  # locale en: milhar ',', decimal '.'
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def _export_xlsx(df: pd.DataFrame, sheet_name: str = "YTG_Mensal") -> bytes:
    bio = io.BytesIO()
    engine = None
    try:
        import xlsxwriter  # noqa: F401
        engine = "xlsxwriter"
    except Exception:
        try:
            import openpyxl  # noqa: F401
            engine = "openpyxl"
        except Exception:
            engine = None
    if engine is None:
        return df.to_csv(index=False).encode("utf-8")
    with pd.ExcelWriter(bio, engine=engine) as xw:
        df.to_excel(xw, index=False, sheet_name=(sheet_name or "Plan1")[:31])
    bio.seek(0)
    return bio.read()

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for c in df.columns:
        lc = str(c).lower().strip()
        if lc in {"familia comercial","família comercial","familia","familia_comercial"}:
            ren[c] = "Família Comercial"
        elif lc in {"indicador","linha","métrica","metrica","descricao","descrição","item","descrição da linha"}:
            ren[c] = "Indicador"
        elif lc in {"mês","mes"}:
            ren[c] = "Mês"
        elif lc == "valor":
            ren[c] = "Valor"
        elif lc == "ano":
            ren[c] = "ano"
    return df.rename(columns=ren)

@st.cache_data(ttl=600, show_spinner=False)
def _load_preco_receita_bruta_long() -> pd.DataFrame:
    if not BASE_CALCULOS_XLSX.exists():
        return pd.DataFrame(columns=["Família Comercial","mes_num","Valor"])
    try:
        raw = pd.read_excel(BASE_CALCULOS_XLSX)
    except Exception:
        return pd.DataFrame(columns=["Família Comercial","mes_num","Valor"])
    df = _normalize_cols(raw.copy())
    if "Indicador" not in df.columns:
        return pd.DataFrame(columns=["Família Comercial","mes_num","Valor"])
    df = df[df["Indicador"].astype(str).str.strip().str.lower() == "receita bruta"].copy()
    if df.empty:
        return pd.DataFrame(columns=["Família Comercial","mes_num","Valor"])
    month_cols = [c for c in df.columns if str(c)[:3].lower() in MESES_MAP_NUM]
    if month_cols:
        id_vars = [c for c in ["Família Comercial"] if c in df.columns]
        mlong = df.melt(id_vars=id_vars, value_vars=month_cols, var_name="Mês", value_name="Valor")
        mlong["mes_num"] = mlong["Mês"].astype(str).str[:3].str.lower().map(MESES_MAP_NUM)
        mlong["Valor"] = pd.to_numeric(mlong["Valor"], errors="coerce").fillna(0.0)
        out = mlong[["Família Comercial","mes_num","Valor"]].copy()
    else:
        if "Valor" not in df.columns:
            num_cands = [c for c in df.columns if c not in {"Família Comercial","Indicador","Mês","mes","ano"} and pd.api.types.is_numeric_dtype(df[c])]
            if not num_cands:
                return pd.DataFrame(columns=["Família Comercial","mes_num","Valor"])
            df = df.rename(columns={num_cands[0]: "Valor"})
        if "Mês" in df.columns:
            mes_num = df["Mês"].astype(str).str[:3].str.lower().map(MESES_MAP_NUM)
        elif "mes" in df.columns:
            mes_num = pd.to_numeric(df["mes"], errors="coerce")
        else:
            return pd.DataFrame(columns=["Família Comercial","mes_num","Valor"])
        out = pd.DataFrame({
            "Família Comercial": df.get("Família Comercial", pd.Series(dtype=str)),
            "mes_num": pd.to_numeric(mes_num, errors="coerce").fillna(0).astype(int),
            "Valor": pd.to_numeric(df["Valor"], errors="coerce").fillna(0.0),
        })
    out = out[(out["mes_num"] >= 1) & (out["mes_num"] <= 12)].copy()
    return out

# ---- genérico para outras linhas (Insumos/Toll/Frete) ----
@st.cache_data(ttl=600, show_spinner=False)
def _load_param_long(indicador_nome: str) -> pd.DataFrame:
    """
    Lê 'base_calculos.xlsx' e devolve long para um Indicador específico:
      ['Família Comercial','mes_num','Valor']
    """
    if not BASE_CALCULOS_XLSX.exists():
        return pd.DataFrame(columns=["Família Comercial","mes_num","Valor"])
    try:
        raw = pd.read_excel(BASE_CALCULOS_XLSX)
    except Exception:
        return pd.DataFrame(columns=["Família Comercial","mes_num","Valor"])
    df = _normalize_cols(raw.copy())
    if "Indicador" not in df.columns:
        return pd.DataFrame(columns=["Família Comercial","mes_num","Valor"])
    df = df[df["Indicador"].astype(str).str.strip().str.lower() == indicador_nome.strip().lower()].copy()
    if df.empty:
        return pd.DataFrame(columns=["Família Comercial","mes_num","Valor"])
    month_cols = [c for c in df.columns if str(c)[:3].lower() in MESES_MAP_NUM]
    if month_cols:
        id_vars = [c for c in ["Família Comercial"] if c in df.columns]
        mlong = df.melt(id_vars=id_vars, value_vars=month_cols, var_name="Mês", value_name="Valor")
        mlong["mes_num"] = mlong["Mês"].astype(str).str[:3].str.lower().map(MESES_MAP_NUM)
        mlong["Valor"] = pd.to_numeric(mlong["Valor"], errors="coerce").fillna(0.0)
        out = mlong[["Família Comercial","mes_num","Valor"]].copy()
    else:
        if "Valor" not in df.columns:
            num_cands = [c for c in df.columns if c not in {"Família Comercial","Indicador","Mês","mes","ano"} and pd.api.types.is_numeric_dtype(df[c])]
            if not num_cands:
                return pd.DataFrame(columns=["Família Comercial","mes_num","Valor"])
            df = df.rename(columns={num_cands[0]: "Valor"})
        if "Mês" in df.columns:
            mes_num = df["Mês"].astype(str).str[:3].str.lower().map(MESES_MAP_NUM)
        elif "mes" in df.columns:
            mes_num = pd.to_numeric(df["mes"], errors="coerce")
        else:
            return pd.DataFrame(columns=["Família Comercial","mes_num","Valor"])
        out = pd.DataFrame({
            "Família Comercial": df.get("Família Comercial", pd.Series(dtype=str)),
            "mes_num": pd.to_numeric(mes_num, errors="coerce").fillna(0).astype(int),
            "Valor": pd.to_numeric(df["Valor"], errors="coerce").fillna(0.0),
        })
    out = out[(out["mes_num"] >= 1) & (out["mes_num"] <= 12)].copy()
    return out

def _preco_dez_por_familia() -> pd.DataFrame:
    base = _load_preco_receita_bruta_long()
    if base.empty:
        return pd.DataFrame(columns=["Família Comercial","Preço Dez (R$/UC)"])
    dez = base[base["mes_num"] == 12].copy()
    if dez.empty:
        return pd.DataFrame(columns=["Família Comercial","Preço Dez (R$/UC)"])
    dez_sorted = dez.sort_values(["Família Comercial"])
    out = (dez_sorted.groupby("Família Comercial", as_index=False)["Valor"]
                     .first()
                     .rename(columns={"Valor": "Preço Dez (R$/UC)"}))
    return out

def _param_dez_por_familia(indicador_nome: str, out_col: str) -> pd.DataFrame:
    base = _load_param_long(indicador_nome)
    if base.empty:
        return pd.DataFrame(columns=["Família Comercial", out_col])
    dez = base[base["mes_num"] == 12].copy()
    if dez.empty:
        return pd.DataFrame(columns=["Família Comercial", out_col])
    dez_sorted = dez.sort_values(["Família Comercial"])
    out = (dez_sorted.groupby("Família Comercial", as_index=False)["Valor"]
           .first()
           .rename(columns={"Valor": out_col}))
    return out

# ------------------------------ Núcleo comum ------------------------------- #
def _volume_ytg_pivot():
    """Retorna (piv_vol, month_names, ano, cutoff)."""
    res_long = re_volume_by_family_long()
    if res_long.empty or not {"Família Comercial","ano","mes","volume"}.issubset(res_long.columns):
        return pd.DataFrame(), [], None, None

    anos = sorted([int(a) for a in res_long["ano"].dropna().unique()])
    if not anos:
        return pd.DataFrame(), [], None, None
    ano = int(anos[-1])

    cut_map = realized_cutoff_by_year(cenario_like="Realizado") or {}
    cutoff = int(cut_map.get(ano, 0))

    ytg = res_long[(res_long["ano"] == ano) & (res_long["mes"] > cutoff)].copy()
    if ytg.empty:
        ytg_months = list(range(cutoff+1, 13)) if cutoff < 12 else []
        piv = pd.DataFrame()
        month_names = [MONTHS_PT.get(m, str(m)) for m in ytg_months]
        return piv, month_names, ano, cutoff

    ytg["mes"] = pd.to_numeric(ytg["mes"], errors="coerce").fillna(0).astype(int)
    ytg["volume"] = pd.to_numeric(ytg["volume"], errors="coerce").fillna(0.0)

    grp = (ytg.groupby(["Família Comercial","mes"], dropna=False)["volume"]
             .sum().reset_index())

    ytg_months = list(range(cutoff + 1, 13)) if cutoff < 12 else []
    piv = grp.pivot_table(index="Família Comercial", columns="mes",
                          values="volume", aggfunc="sum", fill_value=0.0)
    if ytg_months:
        for m in ytg_months:
            if m not in piv.columns:
                piv[m] = 0.0
        piv = piv.reindex(columns=ytg_months)
        month_names = [MONTHS_PT.get(m, str(m)) for m in ytg_months]
        piv.columns = month_names
    else:
        piv = pd.DataFrame(index=piv.index)
        month_names = []
    return piv, month_names, ano, cutoff

def _build_table_from_piv(piv: pd.DataFrame,
                          month_names: list[str],
                          indicador_nome: str,
                          param_col_label: str,
                          value_prefix: str,
                          session_prefix: str,
                          negative: bool) -> pd.DataFrame:
    """
    Monta tabela: Família | Param Dez (R$/UC) | meses + [value_prefix mês (R$)] | Totais.
    negative=True aplica sinal negativo (ex.: Insumos/Toll/Frete).
    Publica totais mensais no session_state com chave f"{session_prefix}_ytg_total_by_month".
    """
    if piv is None or piv.empty:
        return pd.DataFrame()

    df = piv.copy().reset_index()
    # Param Dez por família
    param = _param_dez_por_familia(indicador_nome, param_col_label)
    if not param.empty:
        df = df.merge(param, on="Família Comercial", how="left")
    else:
        df[param_col_label] = 0.0

    # Cálculo mês a mês
    value_cols = []
    param_num = pd.to_numeric(df[param_col_label], errors="coerce").fillna(0.0)
    for mn in month_names:
        base = pd.to_numeric(df[mn], errors="coerce").fillna(0.0)
        val = base * param_num
        if negative:
            val = -val.abs()
        coln = f"{value_prefix} {mn} (R$)"
        df[coln] = val
        value_cols.append(coln)

    # Totais por família
    df["Total YTG"] = df[month_names].sum(axis=1)
    df["Total Volume YTG (UC)"] = df["Total YTG"]
    df[f"Total {value_prefix} YTG (R$)"] = df[value_cols].sum(axis=1) if value_cols else 0.0

    # Ordenação
    cols_order = ["Família Comercial", param_col_label]
    for mn in month_names:
        cols_order.append(mn)
        vc = f"{value_prefix} {mn} (R$)"
        if vc in df.columns:
            cols_order.append(vc)
    cols_order += ["Total YTG", "Total Volume YTG (UC)", f"Total {value_prefix} YTG (R$)"]
    for c in cols_order:
        if c not in df.columns:
            df[c] = 0.0
    df = df[cols_order]

    # Totais globais por mês (soma de famílias)
    tot_by_month = {mn: float(pd.to_numeric(df[f"{value_prefix} {mn} (R$)"], errors="coerce").fillna(0.0).sum())
                    for mn in month_names if f"{value_prefix} {mn} (R$)" in df.columns}
    tot_fy = float(sum(tot_by_month.values()))

    # Publica em session_state
    st.session_state[f"{session_prefix}_ytg_total_by_month"] = tot_by_month
    st.session_state[f"{session_prefix}_ytg_total_fy"] = tot_fy

    # Formatação visual
    df[param_col_label] = df[param_col_label].map(_fmt_6dec_ptbr)
    for c in month_names + value_cols + ["Total YTG", "Total Volume YTG (UC)", f"Total {value_prefix} YTG (R$)"]:
        df[c] = df[c].map(_fmt_6dec_ptbr)

    return df

# ------------------------------ Páginas/Tabelas ------------------------------ #
def build_volume_ytg_mensal_com_preco_dez() -> pd.DataFrame:
    global VOL_YTG_TOTAL_BY_MONTH, RB_YTG_TOTAL_BY_MONTH
    global VOL_YTG_TOTAL_FY, RB_YTG_TOTAL_FY, YTG_MONTHS, YTG_ANO, YTG_CUTOFF
    global INSUMOS_YTG_TOTAL_BY_MONTH, TOLL_YTG_TOTAL_BY_MONTH, FRETE_YTG_TOTAL_BY_MONTH
    global INSUMOS_YTG_TOTAL_FY, TOLL_YTG_TOTAL_FY, FRETE_YTG_TOTAL_FY

    piv, month_names, ano, cutoff = _volume_ytg_pivot()
    if ano is None:
        return pd.DataFrame()

    # RB (preço Dez)
    if piv.empty and month_names:
        cols = ["Família Comercial","Preço Dez (R$/UC)"] + month_names + ["Total YTG","Total Volume YTG (UC)","Total RB YTG (R$)"]
        return pd.DataFrame(columns=cols)

    out = piv.copy()
    out["Total YTG"] = out.sum(axis=1)
    out["Total Volume YTG (UC)"] = out["Total YTG"]

    preco_dez = _preco_dez_por_familia()
    if not preco_dez.empty:
        out = out.merge(preco_dez, left_index=True, right_on="Família Comercial", how="left").set_index("Família Comercial", drop=True)
    else:
        out["Preço Dez (R$/UC)"] = 0.0

    out = out.reset_index()
    rb_cols = []
    if "Preço Dez (R$/UC)" in out.columns:
        preco_series = pd.to_numeric(out["Preço Dez (R$/UC)"], errors="coerce").fillna(0.0)
        for mn in month_names:
            if mn in out.columns:
                out[f"RB {mn} (R$)"] = pd.to_numeric(out[mn], errors="coerce").fillna(0.0) * preco_series
                rb_cols.append(f"RB {mn} (R$)")
    out["Total RB YTG (R$)"] = out[rb_cols].sum(axis=1) if rb_cols else 0.0

    cols_order = ["Família Comercial", "Preço Dez (R$/UC)"]
    for mn in month_names:
        cols_order.append(mn)
        rb_name = f"RB {mn} (R$)"
        if rb_name in out.columns:
            cols_order.append(rb_name)
    cols_order += ["Total YTG", "Total Volume YTG (UC)", "Total RB YTG (R$)"]
    for c in cols_order:
        if c not in out.columns:
            out[c] = 0.0
    out = out[cols_order]

    # Totais globais (RB e Volume) — para reuso
    vol_tot_by_month = {mn: float(pd.to_numeric(out[mn], errors="coerce").fillna(0.0).sum()) for mn in month_names}
    rb_tot_by_month  = {mn: float(pd.to_numeric(out[f"RB {mn} (R$)"], errors="coerce").fillna(0.0).sum()) if f"RB {mn} (R$)" in out.columns else 0.0 for mn in month_names}
    vol_tot_fy = float(sum(vol_tot_by_month.values()))
    rb_tot_fy  = float(sum(rb_tot_by_month.values()))

    YTG_MONTHS = month_names
    YTG_ANO = ano
    YTG_CUTOFF = cutoff
    VOL_YTG_TOTAL_BY_MONTH = vol_tot_by_month
    RB_YTG_TOTAL_BY_MONTH = rb_tot_by_month
    VOL_YTG_TOTAL_FY = vol_tot_fy
    RB_YTG_TOTAL_FY = rb_tot_fy

    st.session_state["ytg_months"] = month_names
    st.session_state["ytg_ano"] = ano
    st.session_state["ytg_cutoff"] = cutoff
    st.session_state["vol_ytg_total_by_month"] = vol_tot_by_month
    st.session_state["rb_ytg_total_by_month"] = rb_tot_by_month
    st.session_state["vol_ytg_total_fy"] = vol_tot_fy
    st.session_state["rb_ytg_total_fy"] = rb_tot_fy

    # Formatação visual (RB)
    for c in month_names + rb_cols + ["Total YTG", "Total Volume YTG (UC)", "Total RB YTG (R$)"]:
        out[c] = out[c].map(_fmt_6dec_ptbr)
    out["Preço Dez (R$/UC)"] = out["Preço Dez (R$/UC)"].map(_fmt_6dec_ptbr)

    # Guarda metadados para as outras tabelas
    st.session_state["_ytg_piv_internal"] = piv  # (númerico)
    st.session_state["_ytg_month_names_internal"] = month_names

    st.caption(f"Ano: **{ano}** · Cutoff (RE): **mês {cutoff}** · YTG = meses > cutoff · Parâmetro Dez = Excel (Indicador → Valor).")
    return out

# --------------------------------- Render ----------------------------------- #
try:
    df_rb = build_volume_ytg_mensal_com_preco_dez()
    if df_rb.empty:
        st.info("Sem dados para montar o YTG mensal. Verifique RES/RE e a planilha de parâmetros.")
    else:
        # Tabela RB
        st.subheader("Receita Bruta — Volume × Preço Dez (R$/UC)")
        st.dataframe(df_rb, use_container_width=True, hide_index=True)

        # Tabelas adicionais: Insumos / Toll / Fretes
        piv = st.session_state.get("_ytg_piv_internal")
        month_names = st.session_state.get("_ytg_month_names_internal", [])

        if piv is not None and not getattr(piv, "empty", True) and month_names:
            st.subheader("Insumos — Volume × Param Dez (R$/UC)")
            df_ins = _build_table_from_piv(
                piv=piv,
                month_names=month_names,
                indicador_nome="Insumos",
                param_col_label="Insumos Dez (R$/UC)",
                value_prefix="Insumos",
                session_prefix="insumos",
                negative=True
            )
            if not df_ins.empty:
                INSUMOS_YTG_TOTAL_BY_MONTH = st.session_state.get("insumos_ytg_total_by_month", {})
                INSUMOS_YTG_TOTAL_FY = st.session_state.get("insumos_ytg_total_fy", 0.0)
                st.dataframe(df_ins, use_container_width=True, hide_index=True)

            st.subheader("Custos Toll Packer — Volume × Param Dez (R$/UC)")
            df_toll = _build_table_from_piv(
                piv=piv,
                month_names=month_names,
                indicador_nome="Custos Toll Packer",
                param_col_label="Toll Dez (R$/UC)",
                value_prefix="Toll",
                session_prefix="toll",
                negative=True
            )
            if not df_toll.empty:
                TOLL_YTG_TOTAL_BY_MONTH = st.session_state.get("toll_ytg_total_by_month", {})
                TOLL_YTG_TOTAL_FY = st.session_state.get("toll_ytg_total_fy", 0.0)
                st.dataframe(df_toll, use_container_width=True, hide_index=True)

            st.subheader("Fretes T1 — Volume × Param Dez (R$/UC)")
            df_fre = _build_table_from_piv(
                piv=piv,
                month_names=month_names,
                indicador_nome="Fretes T1",
                param_col_label="Frete Dez (R$/UC)",
                value_prefix="Frete",
                session_prefix="frete",
                negative=True
            )
            if not df_fre.empty:
                FRETE_YTG_TOTAL_BY_MONTH = st.session_state.get("frete_ytg_total_by_month", {})
                FRETE_YTG_TOTAL_FY = st.session_state.get("frete_ytg_total_fy", 0.0)
                st.dataframe(df_fre, use_container_width=True, hide_index=True)

        # Export RB (principal) — mantém comportamento atual
        xlsx_bytes = _export_xlsx(df_rb, sheet_name="YTG_Mensal_RB")
        st.download_button(
            label="Exportar XLSX (RB)",
            data=xlsx_bytes,
            file_name="volume_ytg_mensal_preco_dez.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=False,
        )

except Exception as e:
    st.error(f"[YTG Mensal + Parâmetros Dez] Erro: {e}")
