# pages/05_DRE_FY_por_Familia.py
# -----------------------------------------------------------------------------
# P&L FY (YTD + YTG) por Família Comercial:
# - Linhas: dicionário canônico do P&L
# - Colunas: cada Família + coluna 'Total' (soma das famílias)
# Extras:
# - Destaque visual (linhas-âncora em cinza + negrito) igual ao DRE.
# - Linhas FIXAS vindas do parquet rateadas PROPORCIONALMENTE ao Volume FY.
# - Subtotais recalculados após a absorção.
# - Exportação para XLSX (fallback p/ CSV).
# -----------------------------------------------------------------------------

from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="DRE FY por Família", page_icon="🧮", layout="wide")

# ---------------------------------------------------------------------
# Imports base do projeto
# ---------------------------------------------------------------------
from core.models import (
    re_volume_by_family_long,
    realized_cutoff_by_year,
)

# =============================================================================
# Loader resiliente do current.parquet
# =============================================================================
try:
    from core.models import load_current_parquet as _models_load_current  # opcional
except Exception:
    _models_load_current = None

def _coerce_cols(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in {"família comercial", "familia comercial", "familia_comercial"}:
            ren[c] = "Família Comercial"
        elif lc == "valor":
            ren[c] = "valor"
        elif lc == "indicador_id":
            ren[c] = "indicador_id"
        elif lc == "cenario":
            ren[c] = "cenario"
        elif lc == "ano":
            ren[c] = "ano"
        elif lc in {"mes", "mês"}:
            ren[c] = "mes"
    if ren:
        df = df.rename(columns=ren)

    for col in ["cenario", "ano", "mes", "indicador_id", "valor"]:
        if col not in df.columns:
            df[col] = pd.Series(dtype="float64" if col == "valor" else "object")

    df["ano"] = pd.to_numeric(df["ano"], errors="coerce").fillna(0).astype(int)
    df["mes"] = pd.to_numeric(df["mes"], errors="coerce").fillna(0).astype(int)
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce").fillna(0.0)

    if "Família Comercial" in df.columns:
        df["Família Comercial"] = df["Família Comercial"].astype(str)
    return df

def load_current_parquet_safe() -> pd.DataFrame:
    if _models_load_current is not None:
        try:
            cur = _models_load_current()
            if cur is not None and not cur.empty:
                return _coerce_cols(cur.copy())
        except Exception:
            pass

    p = Path("data/parquet/current.parquet")
    if not p.exists():
        return pd.DataFrame()
    try:
        cur = pd.read_parquet(p)
    except Exception:
        cur = pd.read_parquet(p, engine="fastparquet")
    return _coerce_cols(cur.copy())

# =============================================================================
# Constantes / Meses / Ordem / Conjuntos
# =============================================================================
MONTHS_PT: dict[int, str] = {
    1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"
}
MESES_NUM = {"jan":1,"fev":2,"mar":3,"abr":4,"mai":5,"jun":6,"jul":7,"ago":8,"set":9,"out":10,"nov":11,"dez":12}

ORDER_CANONICA = [
    "volume_uc",
    "receita_bruta",
    "convenio_impostos",
    "receita_liquida",
    "insumos",
    "custos_toll_packer",
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
    "opex_leao_reg",
    "chargeback",
    "resultado_operacional",
    "depreciacao",
    "ebitda",
]

# Linhas FIXAS (vêm do parquet e DEVEM ser rateadas por volume no FY)
FIXED_ABSORPTION = {
    "estrutura_industrial_variavel",
    "estrutura_industrial_fixa",
    "custos_log_t1_reg",
    "despesas_secundarias_t2",
    "perdas",
    "opex_leao_reg",
    "chargeback",
    "depreciacao",
}

# Linhas-âncora (estilo cinza + negrito)
ANCHORS_BOLD_GRAY = {
    "receita_bruta",
    "receita_liquida",
    "margem_bruta",
    "margem_contribuicao",
    "margem_contribuicao_liquida",
    "resultado_operacional",
    "ebitda",
}

BASE_CALCULOS_XLSX = Path("data/premissas_pnl/base_calculos.xlsx")

# =============================================================================
# Helpers
# =============================================================================
def _norm_txt(s: str) -> str:
    import unicodedata
    s = str(s).strip()
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    return s.lower()

def _load_base_calculos(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_excel(path)
    ren = {}
    for c in df.columns:
        cl = str(c).lower().strip()
        if cl.startswith("fam"): ren[c] = "Família Comercial"
        elif cl in {"mês","mes"}: ren[c] = "Mês"
        elif cl in {"indicador","linha"}: ren[c] = "Indicador"
        elif cl in {"valor","val"}: ren[c] = "Valor"
    return df.rename(columns=ren)

def _param_uc_mensal(params: pd.DataFrame, indicador_alvo: str, col_valor="Valor") -> pd.DataFrame:
    if params.empty:
        return pd.DataFrame(columns=["Família Comercial","mes","valor_uc"])
    df = params.copy()
    if not {"Família Comercial","Indicador","Mês"}.issubset(df.columns):
        return pd.DataFrame(columns=["Família Comercial","mes","valor_uc"])
    df["_indic_n"] = df["Indicador"].map(_norm_txt)
    alvo = _norm_txt(indicador_alvo)
    sub = df[df["_indic_n"] == alvo].copy()
    if sub.empty or col_valor not in sub.columns:
        return pd.DataFrame(columns=["Família Comercial","mes","valor_uc"])
    sub["mes"] = sub["Mês"].astype(str).str[:3].str.lower().map(MESES_NUM)
    sub["mes"] = pd.to_numeric(sub["mes"], errors="coerce").fillna(0).astype(int)
    sub = sub[(sub["mes"]>=1)&(sub["mes"]<=12)].copy()
    sub[col_valor] = pd.to_numeric(sub[col_valor], errors="coerce").fillna(0.0)
    return (sub.groupby(["Família Comercial","mes"], as_index=False)[col_valor]
              .mean().rename(columns={col_valor:"valor_uc"}))

def _price_maps(params: pd.DataFrame):
    rb  = _param_uc_mensal(params, "Receita Bruta").rename(columns={"valor_uc":"preco_rb"})
    ins = _param_uc_mensal(params, "Insumos").rename(columns={"valor_uc":"custo_ins"})
    tol = _param_uc_mensal(params, "Custos Toll Packer").rename(columns={"valor_uc":"custo_toll"})
    fre = _param_uc_mensal(params, "Fretes T1").rename(columns={"valor_uc":"custo_frete"})
    return rb, ins, tol, fre

def _pivot_fy_by_family(monthly_map: dict[tuple[str,str], float], fams: list[str]) -> pd.DataFrame:
    by_fam: dict[tuple[str,str], float] = {}
    for (rid, fam), v in monthly_map.items():
        by_fam[(rid, fam)] = by_fam.get((rid, fam), 0.0) + float(v or 0.0)

    data = []
    pretty = {
        "volume_uc":"Volume (UC)",
        "receita_bruta":"Receita Bruta",
        "convenio_impostos":"Convênio/Impostos",
        "receita_liquida":"Receita Líquida",
        "insumos":"Insumos",
        "custos_toll_packer":"Custos Toll Packer",
        "fretes_t1":"Fretes T1",
        "estrutura_industrial_variavel":"Estrutura Industrial (Var.)",
        "custos_variaveis":"Custos Variáveis",
        "margem_variavel":"Margem Variável",
        "estrutura_industrial_fixa":"Estrutura Industrial (Fixa)",
        "margem_bruta":"Margem Bruta",
        "custos_log_t1_reg":"Custos Log T1 (Reg.)",
        "despesas_secundarias_t2":"Despesas Secundárias T2",
        "perdas":"Perdas",
        "margem_contribuicao":"Margem de Contribuição",
        "dme":"DME",
        "margem_contribuicao_liquida":"Margem Contribuição Líquida",
        "opex_leao_reg":"Opex Leão (Reg.)",
        "chargeback":"Chargeback",
        "resultado_operacional":"Resultado Operacional",
        "depreciacao":"Depreciação",
        "ebitda":"EBITDA",
    }

    for rid in ORDER_CANONICA:
        row = {"indicador_id": rid, "Indicador": pretty.get(rid, rid)}
        total = 0.0
        for f in fams:
            val = float(by_fam.get((rid, f), 0.0))
            row[f] = val
            total += val
        row["Total"] = total
        data.append(row)
    return pd.DataFrame(data)

def _fmt_int(x):
    try:
        return int(np.rint(float(x)))
    except Exception:
        return 0

def _familia_col(df: pd.DataFrame) -> str|None:
    for c in df.columns:
        cl = str(c).lower()
        if cl.startswith("família") or cl.startswith("familia"):
            return c
    return None

def _export_xlsx(df: pd.DataFrame, sheet_name: str = "DRE_FY_Familia") -> bytes:
    import io
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

def _style_anchor_rows(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def row_style(row):
        rid = str(row["indicador_id"])
        if rid in ANCHORS_BOLD_GRAY:
            return ["font-weight: bold; background-color: #f2f2f2"] * len(row)
        return [""] * len(row)
    return df.style.apply(row_style, axis=1)

# =============================================================================
# ABSORÇÃO FY por volume (AJUSTADO)
# =============================================================================
def _absorb_fixed_by_volume(
    df_fy: pd.DataFrame,
    fixed_totals: dict[str, float] | None = None
) -> pd.DataFrame:
    """
    Rateia as linhas FIXED_ABSORPTION proporcionalmente ao Volume FY por família.
    - Se a linha FIXA não existir, cria.
    - Se o 'Total' da linha estiver 0, usa fixed_totals[rid] (quando fornecido).
    - Garante soma das famílias == Total (corrige drift na última família).
    - Recalcula linhas derivadas em FY (convênio, RL, CV, MV, MB, MC, DME, MCL, RO, EBITDA).
    """
    df = df_fy.copy()

    fam_cols = [c for c in df.columns if c not in {"indicador_id", "Indicador", "Total"}]

    # Base de volume FY
    vol_row = df[df["indicador_id"] == "volume_uc"]
    if vol_row.empty:
        return df
    vol = vol_row.iloc[0][fam_cols].astype(float)
    vol_total = float(vol.sum())
    shares = (vol / vol_total) if vol_total > 0 else pd.Series({c: 0.0 for c in fam_cols})

    # pretty map local para possíveis linhas criadas
    pretty = {
        "estrutura_industrial_variavel":"Estrutura Industrial (Var.)",
        "estrutura_industrial_fixa":"Estrutura Industrial (Fixa)",
        "custos_log_t1_reg":"Custos Log T1 (Reg.)",
        "despesas_secundarias_t2":"Despesas Secundárias T2",
        "perdas":"Perdas",
        "opex_leao_reg":"Opex Leão (Reg.)",
        "chargeback":"Chargeback",
        "depreciacao":"Depreciação",
    }

    # Cria linhas FIXAS ausentes e/ou injeta Total FY vindo da fonte
    for rid in FIXED_ABSORPTION:
        mask = df["indicador_id"] == rid
        if not mask.any():
            # cria linha zerada
            new_row = {"indicador_id": rid, "Indicador": pretty.get(rid, rid)}
            for c in fam_cols:
                new_row[c] = 0.0
            new_row["Total"] = 0.0
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            mask = df["indicador_id"] == rid

        # define Total alvo
        tot_df = float(df.loc[mask, "Total"].values[0])
        tot_map = float((fixed_totals or {}).get(rid, 0.0))
        tot_val = tot_map if tot_map > 0 else tot_df  # prioridade ao total vindo do parquet agregado

        # rateia por volume FY
        alloc = (shares * tot_val).round(0).astype(float)
        drift = float(tot_val - alloc.sum())
        if len(fam_cols) > 0:
            alloc.iloc[-1] = alloc.iloc[-1] + drift

        for c in fam_cols:
            df.loc[mask, c] = alloc.get(c, 0.0)
        df.loc[mask, "Total"] = float(alloc.sum())

    # === Recalcula linhas derivadas FY (por família) ===
    def get_row(rid: str) -> pd.Series:
        r = df[df["indicador_id"] == rid]
        if r.empty:
            return pd.Series({c: 0.0 for c in fam_cols})
        return r.iloc[0][fam_cols].astype(float)

    rb   = get_row("receita_bruta")
    conv = -0.256 * rb
    rl   = rb + conv

    ins  = get_row("insumos")
    tol  = get_row("custos_toll_packer")
    fre  = get_row("fretes_t1")
    ei_v = get_row("estrutura_industrial_variavel")
    ei_f = get_row("estrutura_industrial_fixa")

    cv   = ins + tol + fre + ei_v
    mv   = rl + cv
    mb   = mv + ei_f

    t1   = get_row("custos_log_t1_reg")
    t2   = get_row("despesas_secundarias_t2")
    per  = get_row("perdas")
    mc   = mb + (t1 + t2 + per)

    dme  = -0.102 * rl
    mcl  = mc + dme

    opex = get_row("opex_leao_reg")
    cb   = get_row("chargeback")
    rop  = mcl + opex + cb

    dep  = get_row("depreciacao")
    ebt  = rop + dep

    for rid, series in [
        ("convenio_impostos", conv),
        ("receita_liquida", rl),
        ("custos_variaveis", cv),
        ("margem_variavel", mv),
        ("margem_bruta", mb),
        ("margem_contribuicao", mc),
        ("dme", dme),
        ("margem_contribuicao_liquida", mcl),
        ("resultado_operacional", rop),
        ("ebitda", ebt),
    ]:
        if (df["indicador_id"] == rid).any():
            for c in fam_cols:
                df.loc[df["indicador_id"] == rid, c] = series.get(c, 0.0)
        else:
            # cria linha derivada se não existir (robustez)
            new_row = {"indicador_id": rid, "Indicador": rid}
            for c in fam_cols:
                new_row[c] = series.get(c, 0.0)
            new_row["Total"] = float(pd.Series(new_row[c] for c in fam_cols).sum())
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # recomputa Totais por linha
    for i in df.index:
        fam_vals = pd.to_numeric(df.loc[i, fam_cols], errors="coerce").fillna(0.0)
        df.at[i, "Total"] = float(fam_vals.sum())

    return df

# =============================================================================
# Núcleo do cálculo
# =============================================================================
def build_pnl_fy_por_familia():
    cur = load_current_parquet_safe()
    if cur is None or cur.empty:
        st.warning("Parquet atual vazio.")
        return pd.DataFrame(), []

    fam_col = _familia_col(cur)
    if fam_col is None:
        st.warning("Parquet atual sem coluna de Família Comercial.")
        return pd.DataFrame(), []

    anos = sorted([int(a) for a in cur["ano"].dropna().unique()])
    if not anos:
        st.warning("Sem ano no parquet.")
        return pd.DataFrame(), []
    ano = int(anos[-1])
    cutoff_map = realized_cutoff_by_year(cenario_like="Realizado") or {}
    cutoff = int(cutoff_map.get(ano, 0))

    # RES volumes
    res_vol = re_volume_by_family_long()
    if res_vol.empty:
        st.warning("RES volumes vazio.")
        return pd.DataFrame(), []
    res_vol = res_vol[(res_vol["ano"] == ano)].copy()
    res_vol["mes"] = pd.to_numeric(res_vol["mes"], errors="coerce").fillna(0).astype(int)
    res_vol["volume"] = pd.to_numeric(res_vol["volume"], errors="coerce").fillna(0.0)

    # Parametrizações (preços e custos/UC)
    params = _load_base_calculos(BASE_CALCULOS_XLSX)
    rb_map, ins_map, toll_map, fre_map = _price_maps(params)

    # RE cenário
    scen_re = None
    for s in sorted(cur.loc[cur["ano"] == ano, "cenario"].dropna().unique().tolist()):
        ss = str(s).strip().lower()
        if ss.startswith("re"):
            scen_re = s
            break
    if scen_re is None:
        st.warning("Cenário RE não encontrado no parquet.")
        return pd.DataFrame(), []

    # RE por família
    def re_by_family(indic):
        sub = cur[(cur["ano"] == ano) & (cur["cenario"] == scen_re) & (cur["indicador_id"] == indic)]
        if sub.empty or fam_col not in sub.columns:
            return pd.DataFrame(columns=[fam_col, "mes", "valor"])
        g = sub.groupby([fam_col, "mes"])["valor"].sum().reset_index()
        g["mes"] = pd.to_numeric(g["mes"], errors="coerce").fillna(0).astype(int)
        return g

    # RE total por mês (sem família)
    def re_total(indic):
        sub = cur[(cur["ano"] == ano) & (cur["cenario"] == scen_re) & (cur["indicador_id"] == indic)]
        if sub.empty:
            return {m: 0.0 for m in range(1, 13)}
        g = sub.groupby("mes")["valor"].sum()
        return {int(m): float(g.get(m, 0.0)) for m in range(1, 13)}

    # Pré-carrega RE necessários (mês a mês)
    re_rb_fam   = re_by_family("receita_bruta")
    re_ins_fam  = re_by_family("insumos")
    re_tol_fam  = re_by_family("custos_toll_packer")
    re_frt_fam  = re_by_family("fretes_t1")

    re_ei_var_total  = re_total("estrutura_industrial_variavel")
    re_ei_fixa_total = re_total("estrutura_industrial_fixa")
    re_t1_total      = re_total("custos_log_t1_reg")
    re_t2_total      = re_total("despesas_secundarias_t2")
    re_perdas_total  = re_total("perdas")
    re_opex_total    = re_total("opex_leao_reg")
    re_cb_total      = re_total("chargeback")
    re_dep_total     = re_total("depreciacao")

    # Universo de famílias
    fams = sorted(set(res_vol["Família Comercial"].astype(str).unique()))

    # Índices auxiliares
    def to_map(df, val_col):
        if df.empty:
            return {}
        return {(str(r["Família Comercial"]), int(r["mes"])): float(r[val_col]) for _, r in df.iterrows()}

    p_rb   = to_map(rb_map,   "preco_rb")
    p_ins  = to_map(ins_map,  "custo_ins")
    p_tol  = to_map(toll_map, "custo_toll")
    p_fre  = to_map(fre_map,  "custo_frete")

    re_rb_fam_map  = {(str(r[fam_col]), int(r["mes"])): float(r["valor"]) for _, r in re_rb_fam.iterrows()}
    re_ins_fam_map = {(str(r[fam_col]), int(r["mes"])): float(r["valor"]) for _, r in re_ins_fam.iterrows()}
    re_tol_fam_map = {(str(r[fam_col]), int(r["mes"])): float(r["valor"]) for _, r in re_tol_fam.iterrows()}
    re_frt_fam_map = {(str(r[fam_col]), int(r["mes"])): float(r["valor"]) for _, r in re_frt_fam.iterrows()}

    vol_map = {(str(r["Família Comercial"]), int(r["mes"])): float(r["volume"]) for _, r in res_vol.iterrows()}

    # Estrutura de acumulação mensal por família
    M: dict[tuple[str, str], float] = {}  # {(indicador_id, familia) -> soma FY}

    def put(ind, fam, val):
        M[(ind, fam)] = M.get((ind, fam), 0.0) + float(val or 0.0)

    # ---------------- LOOP MENSAL: constroi YTD (RE fam) e YTG (volume×parâmetros) ----------------
    for m in range(1, 13):
        ms = MONTHS_PT[m]
        # Shares para rateio intra-mês (por RB; fallback volume)
        rb_fam_mes = {f: float(re_rb_fam_map.get((f, m), 0.0)) for f in fams}
        sum_rb = sum(rb_fam_mes.values())
        if sum_rb <= 0:
            vol_fam_mes = {f: float(vol_map.get((f, m), 0.0)) for f in fams}
            sum_v = sum(vol_fam_mes.values())
            shares = {f: (vol_fam_mes.get(f, 0.0) / sum_v if sum_v > 0 else 0.0) for f in fams}
        else:
            shares = {f: (rb_fam_mes.get(f, 0.0) / sum_rb) for f in fams}

        if m <= cutoff:
            # YTD (Realizado)
            for f in fams:
                v_re = float(vol_map.get((f, m), 0.0))
                put("volume_uc", f, v_re)
                put("volume_sellout", f, v_re)

            def add_line_from_re(indic, fam_map, total_map, sign=+1.0):
                tot = float(total_map.get(m, 0.0))
                for f in fams:
                    val_f = fam_map.get((f, m), np.nan) if fam_map is not None else np.nan
                    val = float(val_f) if not (isinstance(val_f, float) and np.isnan(val_f)) else shares[f] * tot
                    put(indic, f, sign * val)

            add_line_from_re("receita_bruta", re_rb_fam_map,  re_total("receita_bruta"))
            add_line_from_re("insumos",       re_ins_fam_map, re_total("insumos"))
            add_line_from_re("custos_toll_packer", re_tol_fam_map, re_total("custos_toll_packer"))
            add_line_from_re("fretes_t1",     re_frt_fam_map, re_total("fretes_t1"))

            # FIXAS YTD (rateio do total RE do mês)
            for indic, totmap in [
                ("estrutura_industrial_variavel", re_ei_var_total),
                ("estrutura_industrial_fixa",     re_ei_fixa_total),
                ("custos_log_t1_reg",             re_t1_total),
                ("despesas_secundarias_t2",       re_t2_total),
                ("perdas",                        re_perdas_total),
                ("opex_leao_reg",                 re_opex_total),
                ("chargeback",                    re_cb_total),
                ("depreciacao",                   re_dep_total),
            ]:
                tot = float(totmap.get(m, 0.0))
                for f in fams:
                    put(indic, f, shares[f] * tot)

        else:
            # YTG (Volume × parâmetros por família e mês)
            for f in fams:
                v = float(vol_map.get((f, m), 0.0))
                rb_uc  = float(rb_map.loc[(rb_map["Família Comercial"] == f) & (rb_map["mes"] == m), "preco_rb"].mean()) if not rb_map.empty else 0.0
                ins_uc = float(ins_map.loc[(ins_map["Família Comercial"] == f) & (ins_map["mes"] == m), "custo_ins"].mean()) if not ins_map.empty else 0.0
                tol_uc = float(toll_map.loc[(toll_map["Família Comercial"] == f) & (toll_map["mes"] == m), "custo_toll"].mean()) if not toll_map.empty else 0.0
                fre_uc = float(fre_map.loc[(fre_map["Família Comercial"] == f) & (fre_map["mes"] == m), "custo_frete"].mean()) if not fre_map.empty else 0.0

                put("volume_uc", f, v)
                put("volume_sellout", f, v)
                put("receita_bruta", f, v * rb_uc)
                put("insumos", f, -(abs(ins_uc) * v))
                put("custos_toll_packer", f, -(abs(tol_uc) * v))
                put("fretes_t1", f, -(abs(fre_uc) * v))

            # FIXAS YTG: mantém total do RE do mês e rateia por share do mês (rb novo; fallback volume)
            rb_new_fam = {f: M.get(("receita_bruta", f), 0.0) for f in fams}
            # (para robustez, se tudo zero, volta a shares por volume)
            denom_rb = sum(rb_new_fam.values())
            if denom_rb <= 0:
                denom_v = sum([vol_map.get((ff, m), 0.0) for ff in fams])
                shares_y = {f: (vol_map.get((f, m), 0.0) / denom_v if denom_v > 0 else 0.0) for f in fams}
            else:
                shares_y = {f: (rb_new_fam[f] / denom_rb) for f in fams}

            for indic, totmap in [
                ("estrutura_industrial_variavel", re_ei_var_total),
                ("estrutura_industrial_fixa",     re_ei_fixa_total),
                ("custos_log_t1_reg",             re_t1_total),
                ("despesas_secundarias_t2",       re_t2_total),
                ("perdas",                        re_perdas_total),
                ("opex_leao_reg",                 re_opex_total),
                ("chargeback",                    re_cb_total),
                ("depreciacao",                   re_dep_total),
            ]:
                tot = float(totmap.get(m, 0.0))
                for f in fams:
                    put(indic, f, shares_y[f] * tot)

    # Monta FY por família
    df_fy_num = _pivot_fy_by_family(M, fams)

    # === Totais FY das linhas FIXAS vindas do parquet (RE mês a mês somado) ===
    fixed_totals_fy = {
        "estrutura_industrial_variavel": float(sum(re_ei_var_total.values())),
        "estrutura_industrial_fixa":     float(sum(re_ei_fixa_total.values())),
        "custos_log_t1_reg":             float(sum(re_t1_total.values())),
        "despesas_secundarias_t2":       float(sum(re_t2_total.values())),
        "perdas":                        float(sum(re_perdas_total.values())),
        "opex_leao_reg":                 float(sum(re_opex_total.values())),
        "chargeback":                    float(sum(re_cb_total.values())),
        "depreciacao":                   float(sum(re_dep_total.values())),
    }

    # ABSORÇÃO FY (linhas fixas) + RECÁLCULO subtotais FY
    df_fy_num = _absorb_fixed_by_volume(df_fy_num, fixed_totals=fixed_totals_fy)

    # Conversão para int na visão
    df_fy_view = df_fy_num.copy()
    for c in df_fy_view.columns:
        if c not in {"indicador_id", "Indicador"}:
            df_fy_view[c] = df_fy_view[c].map(_fmt_int)

    st.caption(
        f"FY = YTD + YTG · Cutoff (RE): **mês {cutoff}** · "
        f"Linhas FIXAS rateadas pelo **share de Volume FY** por família. Subtotais recalculados."
    )

    return df_fy_view, fams

# =============================================================================
# Render
# =============================================================================
try:
    df, _familias = build_pnl_fy_por_familia()
    if df.empty:
        st.info("Sem dados para montar o P&L FY por Família.")
    else:
        styled = _style_anchor_rows(df)
        st.dataframe(styled, use_container_width=True, hide_index=True)

        xlsx_bytes = _export_xlsx(df, sheet_name="DRE_FY_Familia")
        st.download_button(
            "Exportar XLSX",
            data=xlsx_bytes,
            file_name="dre_fy_por_familia.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=False,
        )
except Exception as e:
    st.error(f"[DRE FY por Família] Erro: {e}")
