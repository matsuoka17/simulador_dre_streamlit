# core/rolagem_estoque.py
# --------------------------------------------------------------------------------------
# Rolagem de Estoque — COMPLETO (ajustado para:
#  - estoque_inicial.xlsx LONG por SKU  -> consolida por Família Comercial
#  - vol_prod_ytg.xlsx WIDE com colunas 'YYYY-MM' (com ou sem espaços) -> derrete p/ long
#  - também aceita 'ano'+'mes' ou coluna única 'aaaa-mm'
#  - VENDAS usa gateway idêntico ao 03_DRE: prefer_ui -> UI (wide/edit) com bootstrap; OFF -> baseline
#  - Estoque Final permite NEGATIVO (sem clamp)
# --------------------------------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import pandas as pd
import numpy as np

# --------------------------------- Paths ------------------------------------- #
DATA_DIR = Path("data")
PARQUET_DIR = DATA_DIR / "parquet"

VOL_PROD_YTG_XLSX    = DATA_DIR / "vol_prod_ytg" / "vol_prod_ytg.xlsx"
ESTOQUE_INICIAL_XLSX = DATA_DIR / "estoque"      / "estoque_inicial.xlsx"
RES_WORKING_PARQUET  = PARQUET_DIR / "res_working.parquet"

# ------------------------------ Constantes ----------------------------------- #
MONTH_MAP_PT: Dict[int, str] = {
    1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"
}

# --------------------------------- Utils ------------------------------------- #
def _as_series(x):
    return x if isinstance(x, pd.Series) else pd.Series(x)

def _coerce_int(s, default=0) -> pd.Series:
    s = _as_series(s)
    return pd.to_numeric(s, errors="coerce").fillna(default).astype("Int64")

def _coerce_float(s, default=0.0) -> pd.Series:
    s = _as_series(s)
    return pd.to_numeric(s, errors="coerce").fillna(default).astype(float)

def _col_familia(df: pd.DataFrame) -> str:
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in {"família comercial","familia comercial","familia","familia_comercial"}:
            return c
    if "Família Comercial" in df.columns:
        return "Família Comercial"
    return df.columns[0]

def _ensure_cols(df: pd.DataFrame, cols: Dict[str, float | int]) -> pd.DataFrame:
    out = df.copy()
    for k, v in cols.items():
        if k not in out.columns:
            out[k] = v
    return out

# -------------------------------- Calendário YTG ----------------------------- #
def _detect_wide_ym_columns(df: pd.DataFrame) -> List[str]:
    """
    Detecta colunas no padrão 'YYYY-MM' (permitindo espaços antes/depois).
    """
    ym_cols = []
    for c in df.columns:
        s = str(c).strip()
        if len(s) == 7 and s[4] == "-" and s[:4].isdigit() and s[5:7].isdigit():
            ym_cols.append(c)
    return ym_cols

def load_vol_prod_ytg_calendar_audit() -> Tuple[pd.DataFrame, dict]:
    """
    Lê vol_prod_ytg.xlsx e retorna calendário + famílias:
      ['Família Comercial','ano','mes']
    Aceita:
      - 'ano'/'mes'
      - coluna única 'aaaa-mm'
      - WIDE: colunas 'YYYY-MM' (com ou sem espaços)
    """
    audit = {
        "file_exists": VOL_PROD_YTG_XLSX.exists(),
        "raw_rows": 0,
        "detected_cols": [],
        "calendar_mode": None,
        "anos_detectados": [],
        "meses_detectados": [],
        "familias_detectadas": 0,
        "sample": None,
        "messages": [],
    }

    if not audit["file_exists"]:
        audit["messages"].append(f"Arquivo não encontrado: {VOL_PROD_YTG_XLSX}")
        return pd.DataFrame(columns=["Família Comercial","ano","mes"]), audit

    df = pd.read_excel(VOL_PROD_YTG_XLSX)
    audit["raw_rows"] = len(df)
    audit["detected_cols"] = [str(c) for c in df.columns]

    if df.empty:
        audit["messages"].append("vol_prod_ytg.xlsx está vazio.")
        return pd.DataFrame(columns=["Família Comercial","ano","mes"]), audit

    fam = _col_familia(df)
    df = df.rename(columns={fam: "Família Comercial"})

    # 1) ano/mes
    if "ano" in df.columns and "mes" in df.columns:
        audit["calendar_mode"] = "ano_mes_separados"
        out = df[{"Família Comercial","ano","mes"}].copy()
        out["ano"] = _coerce_int(out["ano"])
        out["mes"] = _coerce_int(out["mes"])

    else:
        # 2) coluna única 'aaaa-mm'
        ym_col = None
        for c in df.columns:
            if df[c].astype(str).str.strip().str.match(r"^\d{4}-\d{1,2}$", na=False).any():
                ym_col = c; break

        if ym_col is not None:
            audit["calendar_mode"] = "aaaa-mm(coluna_unica)"
            s = df[ym_col].astype(str).str.strip()
            m = s.str.match(r"^\d{4}-\d{1,2}$", na=False)
            sub = df[m].copy()
            sub["ano"] = pd.to_numeric(sub[ym_col].str.slice(0,4), errors="coerce").astype("Int64")
            sub["mes"] = pd.to_numeric(sub[ym_col].str.slice(5), errors="coerce").astype("Int64")
            out = sub[["Família Comercial","ano","mes"]]

        else:
            # 3) formato WIDE: colunas 'YYYY-MM' (com espaços)
            ym_cols = _detect_wide_ym_columns(df)
            if ym_cols:
                audit["calendar_mode"] = "WIDE(YYYY-MM)"
                # derreter p/ long
                rows = []
                for c in ym_cols:
                    s = str(c).strip()
                    ano = int(s[:4]); mes = int(s[5:7])
                    tmp = df[["Família Comercial", c]].copy()
                    tmp["ano"] = ano; tmp["mes"] = mes
                    rows.append(tmp.drop(columns=[c]))
                out = pd.concat(rows, ignore_index=True)
            else:
                audit["messages"].append("Não encontrei colunas de calendário ('ano'/'mes', 'aaaa-mm' ou WIDE 'YYYY-MM').")
                return pd.DataFrame(columns=["Família Comercial","ano","mes"]), audit

    # Sanitiza
    out = out.dropna(subset=["Família Comercial","ano","mes"])
    out["ano"] = _coerce_int(out["ano"])
    out["mes"] = _coerce_int(out["mes"])
    out = out[(out["mes"] >= 1) & (out["mes"] <= 12)]

    if out.empty:
        audit["messages"].append("Após sanitização, calendário ficou vazio.")
        return pd.DataFrame(columns=["Família Comercial","ano","mes"]), audit

    audit["anos_detectados"]     = sorted(out["ano"].dropna().astype(int).unique().tolist())
    audit["meses_detectados"]    = sorted(out["mes"].dropna().astype(int).unique().tolist())
    audit["familias_detectadas"] = int(out["Família Comercial"].nunique())
    audit["sample"]              = out.head(5).to_dict(orient="records")
    return out, audit

def load_vol_prod_ytg() -> pd.DataFrame:
    """
    Produção YTG em LONG:
      ['Família Comercial','ano','mes','producao']
    Aceita:
      - 'ano'/'mes' + coluna de produção
      - 'aaaa-mm' + produção
      - WIDE com colunas 'YYYY-MM' (com ou sem espaços), valores = produção
    """
    if not VOL_PROD_YTG_XLSX.exists():
        return pd.DataFrame(columns=["Família Comercial","ano","mes","producao"])

    df = pd.read_excel(VOL_PROD_YTG_XLSX)
    if df.empty:
        return pd.DataFrame(columns=["Família Comercial","ano","mes","producao"])

    fam = _col_familia(df)
    df = df.rename(columns={fam: "Família Comercial"})

    # 1) ano/mes
    if "ano" in df.columns and "mes" in df.columns:
        df["ano"] = _coerce_int(df["ano"])
        df["mes"] = _coerce_int(df["mes"])
        # detectar coluna de produção
        prod_col = None
        for c in df.columns:
            lc = str(c).lower().strip()
            if lc in {"producao","produção","vol_prod","vol_producao","volume_producao","volume","qtd"}:
                prod_col = c; break
        if prod_col is None:
            cand = [c for c in df.columns if c not in {"Família Comercial","ano","mes"}]
            prod_col = cand[0] if cand else None
        df["producao"] = _coerce_float(df[prod_col]) if prod_col else 0.0
        out = df[["Família Comercial","ano","mes","producao"]]

    else:
        # 2) coluna única 'aaaa-mm'
        ym_col = None
        for c in df.columns:
            if df[c].astype(str).str.strip().str.match(r"^\d{4}-\d{1,2}$", na=False).any():
                ym_col = c; break

        if ym_col is not None:
            s = df[ym_col].astype(str).str.strip()
            m = s.str.match(r"^\d{4}-\d{1,2}$", na=False)
            sub = df[m].copy()
            sub["ano"] = pd.to_numeric(sub[ym_col].str.slice(0,4), errors="coerce").astype("Int64")
            sub["mes"] = pd.to_numeric(sub[ym_col].str.slice(5), errors="coerce").astype("Int64")
            # detectar produção
            prod_col = None
            for c in sub.columns:
                if c in {"Família Comercial", ym_col, "ano", "mes"}:
                    continue
                prod_col = c; break
            sub["producao"] = _coerce_float(sub[prod_col]) if prod_col else 0.0
            out = sub[["Família Comercial","ano","mes","producao"]]

        else:
            # 3) WIDE 'YYYY-MM' (com ou sem espaços) — valores = produção
            ym_cols = _detect_wide_ym_columns(df)
            if not ym_cols:
                return pd.DataFrame(columns=["Família Comercial","ano","mes","producao"])
            rows = []
            for c in ym_cols:
                s = str(c).strip()
                ano = int(s[:4]); mes = int(s[5:7])
                tmp = df[["Família Comercial", c]].copy()
                tmp = tmp.rename(columns={c: "producao"})
                tmp["ano"] = ano; tmp["mes"] = mes
                rows.append(tmp[["Família Comercial","ano","mes","producao"]])
            out = pd.concat(rows, ignore_index=True)
            out["producao"] = _coerce_float(out["producao"])

    out = out.dropna(subset=["Família Comercial","ano","mes"])
    out["ano"] = _coerce_int(out["ano"])
    out["mes"] = _coerce_int(out["mes"])
    out = out[(out["mes"]>=1) & (out["mes"]<=12)]
    return out[["Família Comercial","ano","mes","producao"]]

# ---------------- Estoque inicial LONG por SKU → Família (com auditoria) ----- #
def load_estoque_inicial_audit() -> Tuple[pd.DataFrame, dict]:
    """
    Lê data/estoque/estoque_inicial.xlsx no formato LONG por SKU e consolida por Família Comercial.
    Saída:
      df_out: ['Família Comercial','estoque_inicial'] (agregado)
      audit : diagnósticos
    """
    audit = {
        "file_exists": ESTOQUE_INICIAL_XLSX.exists(),
        "raw_rows": 0,
        "detected_cols": [],
        "col_estoque_detectada": None,
        "familias_detectadas_raw": 0,
        "familias_apos_agregacao": 0,
        "sample_raw": None,
        "sample_out": None,
        "messages": [],
    }

    if not audit["file_exists"]:
        audit["messages"].append(f"Arquivo não encontrado: {ESTOQUE_INICIAL_XLSX}")
        return pd.DataFrame(columns=["Família Comercial","estoque_inicial"]), audit

    df_raw = pd.read_excel(ESTOQUE_INICIAL_XLSX)
    audit["raw_rows"] = len(df_raw)
    audit["detected_cols"] = [str(c) for c in df_raw.columns]

    if df_raw.empty:
        audit["messages"].append("estoque_inicial.xlsx está vazio.")
        return pd.DataFrame(columns=["Família Comercial","estoque_inicial"]), audit

    fam_col = _col_familia(df_raw)
    df = df_raw.rename(columns={fam_col: "Família Comercial"}).copy()

    # detectar a coluna de estoque (e.g., "Estoque inicial Ago UC")
    estoque_col = None
    for c in df.columns:
        lc = str(c).lower().strip()
        if "estoque" in lc and "inicial" in lc:
            estoque_col = c; break
    if estoque_col is None:
        for c in df.columns:
            lc = str(c).lower().strip()
            if lc.startswith("estoque"):
                estoque_col = c; break
    audit["col_estoque_detectada"] = estoque_col

    if estoque_col is None:
        audit["messages"].append("Não encontrei coluna de estoque inicial na planilha.")
        return pd.DataFrame(columns=["Família Comercial","estoque_inicial"]), audit

    df["estoque_inicial"] = _coerce_float(df[estoque_col])

    # Agregar por Família Comercial
    audit["familias_detectadas_raw"] = int(df["Família Comercial"].nunique())
    out = (df.groupby("Família Comercial", as_index=False)["estoque_inicial"]
             .sum())
    audit["familias_apos_agregacao"] = int(out["Família Comercial"].nunique())

    audit["sample_raw"] = df_raw.head(5).to_dict(orient="records")
    audit["sample_out"] = out.head(5).to_dict(orient="records")

    return out[["Família Comercial","estoque_inicial"]], audit

# ------------------------ Calendário a partir do YTG ------------------------- #
def _calendar_from_ytg(prod: pd.DataFrame) -> Tuple[int, List[int], List[str]]:
    """Retorna (ano, months, fams) a partir do YTG (último ano disponível)."""
    if prod.empty:
        year = pd.Timestamp.today().year
        return year, [], []
    years = pd.to_numeric(prod["ano"], errors="coerce").dropna().astype(int)
    year = int(years.max())
    sub = prod[prod["ano"] == year].copy()
    months = sorted(pd.to_numeric(sub["mes"], errors="coerce").dropna().astype(int).unique().tolist())
    fams = sorted(sub["Família Comercial"].dropna().astype(str).unique().tolist())
    return year, months, fams

# --------------------- Tabela Inicial (somente estoque) ---------------------- #
def build_tabela_inicial_with_audit(
    df_vol_prod_ytg: Optional[pd.DataFrame] = None,
    df_estoque_inicial: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Constrói a base (Família × meses do YTG) e injeta SOMENTE o estoque inicial
    no primeiro mês do plano. Retorna (df, audit_info).
    Saída df (LONG):
      ['Família Comercial','ano','mes','mes_nome','estoque_inicial']
    """
    audit = {"steps": [], "calendar": {}, "estoque_inicial": {}, "merge": {}, "messages": []}

    # Calendário do YTG
    if df_vol_prod_ytg is None:
        cal, cal_audit = load_vol_prod_ytg_calendar_audit()
    else:
        cal = df_vol_prod_ytg.copy()
        cal_audit = {"messages": ["Calendário recebido por parâmetro."], "raw_rows": len(cal)}
    audit["calendar"] = cal_audit

    if cal.empty:
        audit["messages"].append("Calendário YTG vazio — não há como construir a grade.")
        return pd.DataFrame(columns=["Família Comercial","ano","mes","mes_nome","estoque_inicial"]), audit

    years = cal["ano"].dropna().astype(int).unique()
    year = int(years.max())
    months = sorted(cal.loc[cal["ano"]==year, "mes"].dropna().astype(int).unique().tolist())
    fams = sorted(cal.loc[cal["ano"]==year, "Família Comercial"].dropna().astype(str).unique().tolist())
    audit["steps"].append(f"Ano do plano: {year}; Meses: {months}; Famílias: {len(fams)}")

    if not months or not fams:
        audit["messages"].append("Meses ou famílias não detectados para o ano selecionado.")
        return pd.DataFrame(columns=["Família Comercial","ano","mes","mes_nome","estoque_inicial"]), audit

    first_month = int(min(months))

    # Estoque Inicial (agregado por Família)
    if df_estoque_inicial is None:
        ini_agg, ini_audit = load_estoque_inicial_audit()
    else:
        ini_agg = df_estoque_inicial.copy()
        ini_audit = {"messages": ["Estoque inicial recebido por parâmetro."], "raw_rows": len(ini_agg)}
    audit["estoque_inicial"] = ini_audit

    if ini_agg.empty:
        ini_map = pd.DataFrame({
            "Família Comercial": fams,
            "ano": year,
            "mes": first_month,
            "estoque_inicial": 0.0
        })
        audit["steps"].append("Estoque inicial ausente — preenchido com zeros no 1º mês para todas as famílias.")
    else:
        ini_map = ini_agg.copy()
        ini_map["ano"] = year
        ini_map["mes"] = first_month
        # garantir cobertura de todas as famílias do plano
        missing = set(fams) - set(ini_map["Família Comercial"].astype(str))
        if missing:
            ini_map = pd.concat([
                ini_map,
                pd.DataFrame({"Família Comercial": sorted(list(missing)),
                              "ano": year, "mes": first_month, "estoque_inicial": 0.0})
            ], ignore_index=True)
            audit["steps"].append(f"Complementei estoque inicial com 0 para {len(missing)} família(s) faltantes.")

    ini_map["estoque_inicial"] = _coerce_float(ini_map["estoque_inicial"])

    # Universo Família × Meses
    base = pd.MultiIndex.from_product([fams, [year], months],
                                      names=["Família Comercial","ano","mes"]).to_frame(index=False)
    audit["merge"]["universe_rows"] = len(base)

    # Merge (apenas estoque inicial)
    out = base.merge(ini_map, on=["Família Comercial","ano","mes"], how="left")
    out["estoque_inicial"] = _coerce_float(out["estoque_inicial"]).fillna(0.0)
    out["mes_nome"] = out["mes"].map(MONTH_MAP_PT)
    out = out.sort_values(["Família Comercial","ano","mes"]).reset_index(drop=True)

    audit["merge"]["final_rows"]   = len(out)
    audit["merge"]["familias_out"] = int(out["Família Comercial"].nunique())
    audit["merge"]["months_out"]   = sorted(out["mes"].dropna().astype(int).unique().tolist())
    audit["sample_out"]            = out.head(10).to_dict(orient="records")

    return out[["Família Comercial","ano","mes","mes_nome","estoque_inicial"]], audit

# ------------------------------ Vendas (UI/RES) ------------------------------ #
def _ui_month_map() -> Dict[str, int]:
    m = {
        "jan":1, "janeiro":1, "jan.":1,
        "fev":2, "fevereiro":2, "fev.":2,
        "mar":3, "março":3, "marco":3, "mar.":3,
        "abr":4, "abril":4, "abr.":4,
        "mai":5, "maio":5, "mai.":5,
        "jun":6, "junho":6, "jun.":6,
        "jul":7, "julho":7, "jul.":7,
        "ago":8, "agosto":8, "ago.":8,
        "set":9, "setembro":9, "set.":9, "sep":9,
        "out":10, "outubro":10, "out.":10, "oct":10,
        "nov":11, "novembro":11, "nov.":11,
        "dez":12, "dezembro":12, "dez.":12, "dec":12,
    }
    m.update({k.capitalize(): v for k, v in m.items()})
    return m

def _normalize_ui_wide_to_long(df_wide: pd.DataFrame, ano: int) -> pd.DataFrame:
    if df_wide is None or len(df_wide) == 0:
        return pd.DataFrame(columns=["Família Comercial","ano","mes","vendas"])
    df = df_wide.copy()
    if "Família Comercial" not in df.columns:
        df = df.reset_index().rename(columns={df.columns[0]:"Família Comercial"})
    mm = _ui_month_map()
    rows = []
    inv_month = {v:k for k,v in MONTH_MAP_PT.items()}
    for col in df.columns:
        if col == "Família Comercial": continue
        key = str(col).strip()
        mes = mm.get(key.lower()) or inv_month.get(key)
        if mes is None:
            continue
        tmp = df[["Família Comercial", col]].copy()
        tmp["ano"] = int(ano); tmp["mes"] = int(mes)
        tmp = tmp.rename(columns={col:"vendas"})
        rows.append(tmp)
    if not rows:
        return pd.DataFrame(columns=["Família Comercial","ano","mes","vendas"])
    out = pd.concat(rows, ignore_index=True)
    out["vendas"] = _coerce_float(out["vendas"])
    return out[["Família Comercial","ano","mes","vendas"]]

def _filter_to_volume_uc_long(df: pd.DataFrame, ano: int) -> pd.DataFrame:
    """Uniformiza pra Volume UC (elimina sellout/outros), mesmo padrão usado na DRE."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Família Comercial","ano","mes","vendas"])
    out = df.copy()

    # Normaliza nome de família
    fam_col = next((c for c in out.columns if str(c).lower().strip() in
                   {"família comercial","familia comercial","familia","familia_comercial"}), "Família Comercial")
    if fam_col != "Família Comercial":
        out = out.rename(columns={fam_col: "Família Comercial"})

    # Se houver coluna de tipo/métrica, filtra UC
    type_cols = [c for c in out.columns if str(c).lower().strip() in
                 {"metric","tipo","tipo_volume","medida","tipo de volume"}]
    if type_cols:
        tcol = type_cols[0]
        out[tcol] = out[tcol].astype(str).str.lower().str.strip()
        uc_keys = {"uc","volume_uc","volume uc","vol_uc"}
        out = out[out[tcol].isin(uc_keys)]

    # Detecta coluna numérica de UC se não for 'vendas'
    if "vendas" not in out.columns:
        num_cols = [c for c in out.columns
                    if c not in {"Família Comercial","ano","mes"} and pd.api.types.is_numeric_dtype(out[c])]
        uc_num = [c for c in num_cols if "uc" in str(c).lower()]
        if uc_num:
            out = out.rename(columns={uc_num[0]: "vendas"})
        elif num_cols:
            out = out.rename(columns={num_cols[0]: "vendas"})

    out["ano"] = _coerce_int(out["ano"])
    out["mes"] = _coerce_int(out["mes"])
    out["vendas"] = _coerce_float(out["vendas"])
    out = out[out["ano"] == int(ano)]
    out = out.groupby(["Família Comercial","ano","mes"], as_index=False)["vendas"].sum()
    return out[["Família Comercial","ano","mes","vendas"]]

def load_vendas_ui(ano: int) -> pd.DataFrame:
    """
    Mantida para compatibilidade, mas o consumo oficial deve ser via get_vendas_uc_long().
    """
    try:
        import streamlit as st
        vendas_long = pd.DataFrame(columns=["Família Comercial","ano","mes","vendas"])
        # volumes_wide[ano]
        if "volumes_wide" in st.session_state:
            wide_by_year = st.session_state["volumes_wide"]
            if isinstance(wide_by_year, dict) and ano in wide_by_year:
                df_wide = pd.DataFrame(wide_by_year[ano])
                vend_wide = _normalize_ui_wide_to_long(df_wide, ano)
                vend_wide = _filter_to_volume_uc_long(vend_wide, ano)
                vendas_long = pd.concat([vendas_long, vend_wide], ignore_index=True)
        # volumes_edit long
        if "volumes_edit" in st.session_state:
            ve = pd.DataFrame(st.session_state["volumes_edit"])
            if not ve.empty:
                ve = _filter_to_volume_uc_long(ve, ano)
                vendas_long = pd.concat([vendas_long, ve], ignore_index=True)

        if vendas_long.empty:
            return vendas_long
        vendas_long = vendas_long.groupby(["Família Comercial","ano","mes"], as_index=False)["vendas"].sum()
        return vendas_long
    except Exception:
        return pd.DataFrame(columns=["Família Comercial","ano","mes","vendas"])

def load_vendas_res(ano: int, meses: List[int]) -> pd.DataFrame:
    if not RES_WORKING_PARQUET.exists():
        return pd.DataFrame(columns=["Família Comercial","ano","mes","vendas"])
    df = pd.read_parquet(RES_WORKING_PARQUET)
    if df.empty:
        return pd.DataFrame(columns=["Família Comercial","ano","mes","vendas"])

    # Normaliza família
    fam = _col_familia(df)
    df = df.rename(columns={fam:"Família Comercial"})

    # Detecta ano/mes
    if "ano" not in df.columns or "mes" not in df.columns:
        ym_col = None
        for c in df.columns:
            if df[c].dtype == object and df[c].astype(str).str.match(r"^\d{4}-\d{1,2}$", na=False).any():
                ym_col = c; break
        if ym_col is not None:
            df["ano"] = pd.to_numeric(df[ym_col].astype(str).str.slice(0,4), errors="coerce")
            df["mes"] = pd.to_numeric(df[ym_col].astype(str).str.slice(5), errors="coerce")

    # Filtro UC / coluna numérica
    df = _filter_to_volume_uc_long(df, ano)

    if df.empty:
        return df

    df = df[df["mes"].isin([int(m) for m in meses])]
    out = df.groupby(["Família Comercial","ano","mes"], as_index=False)["vendas"].sum()
    return out

# --------------------- GATEWAY idêntico ao 03_DRE (vendas UC) ---------------- #
def _bootstrap_ui_volumes_from_res(load_vendas_res_func, year: int, meses: List[int]) -> pd.DataFrame:
    """
    Bootstrap 1x: popula volumes_edit (LONG) a partir do baseline (res_working),
    igual ao 03_DRE — apenas quando a UI estiver vazia.
    """
    try:
        import streamlit as st
    except Exception:
        return pd.DataFrame()

    # Se já existe UI, não faz nada
    if "volumes_wide" in st.session_state:
        wide = st.session_state["volumes_wide"]
        if isinstance(wide, dict) and year in wide and not pd.DataFrame(wide[year]).empty:
            return pd.DataFrame()
    if "volumes_edit" in st.session_state and not pd.DataFrame(st.session_state["volumes_edit"]).empty:
        return pd.DataFrame()

    vend_res = load_vendas_res_func(year, meses)  # LONG baseline (já normalizado em UC)
    if vend_res.empty:
        return pd.DataFrame()

    # Escreve no session_state para a UI (padrão DRE)
    st.session_state["volumes_edit"] = vend_res.copy()
    return vend_res

def get_vendas_uc_long(ano: int, meses: List[int], prefer_ui: bool = True) -> Tuple[pd.DataFrame, str]:
    """
    GATEWAY ÚNICO — igual à DRE:
      - prefer_ui=True  -> UI (volumes_wide[ano] -> volumes_edit) ou bootstrap 1x se vazio.
      - prefer_ui=False -> baseline (res_working) SEM bootstrap.
    Retorna: (DataFrame LONG [Família Comercial, ano, mes, vendas], source_str)
    """
    source = "none"
    try:
        import streamlit as st
    except Exception:
        # Sem Streamlit (ex.: rodando batch), cai no baseline
        base = load_vendas_res(ano, meses)
        return base, "Baseline (res_working/batch)"

    if prefer_ui:
        # 1) UI wide
        if "volumes_wide" in st.session_state:
            wide = st.session_state["volumes_wide"]
            if isinstance(wide, dict) and ano in wide:
                dfw = pd.DataFrame(wide[ano])
                if not dfw.empty:
                    vend_long = _normalize_ui_wide_to_long(dfw, ano)
                    vend_long = _filter_to_volume_uc_long(vend_long, ano)
                    vend_long = vend_long[vend_long["mes"].isin(meses)]
                    source = "UI (wide)"
                    return vend_long, source

        # 2) UI long (volumes_edit)
        if "volumes_edit" in st.session_state:
            ve = pd.DataFrame(st.session_state["volumes_edit"])
            if not ve.empty:
                ve = _filter_to_volume_uc_long(ve, ano)
                ve = ve[ve["mes"].isin(meses)]
                source = "UI (edit)"
                return ve, source

        # 3) Bootstrap 1x (puxa baseline -> grava volumes_edit)
        _bootstrap_ui_volumes_from_res(load_vendas_res, ano, meses)
        if "volumes_edit" in st.session_state:
            ve = pd.DataFrame(st.session_state["volumes_edit"])
            ve = _filter_to_volume_uc_long(ve, ano)
            ve = ve[ve["mes"].isin(meses)]
            source = "Bootstrap (baseline→UI)"
            return ve, source

        return pd.DataFrame(columns=["Família Comercial","ano","mes","vendas"]), source

    # prefer_ui=False -> baseline (res_working) cru, sem bootstrap
    vend_res = load_vendas_res(ano, meses)
    out = vend_res[vend_res["mes"].isin(meses)] if not vend_res.empty else vend_res
    source = "Baseline (res_working)"
    return out, source

# --------------------------- Rolagem (LONG e WIDE) --------------------------- #
def _calendar_from_production_or_audit() -> Tuple[int, List[int], List[str]]:
    ytg = load_vol_prod_ytg()
    year, months, fams = _calendar_from_ytg(ytg)
    return year, months, fams

def build_rolagem_estoque_base(
    prefer_ui_for_sales: bool = True,
    df_vol_prod_ytg: Optional[pd.DataFrame] = None,
    df_estoque_inicial_agg: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    LONG essencial:
      ['Família Comercial','ano','mes','mes_nome','estoque_inicial','producao','vendas','estoque_final']
    - Vendas: usa get_vendas_uc_long (gateway da DRE).
    - Estoque Final permite NEGATIVO (sem clamp).
    """
    # Bases
    ytg = df_vol_prod_ytg if df_vol_prod_ytg is not None else load_vol_prod_ytg()
    year, months, fams = _calendar_from_ytg(ytg)
    if not months:
        return pd.DataFrame(columns=[
            "Família Comercial","ano","mes","mes_nome",
            "estoque_inicial","producao","vendas","estoque_final"
        ])
    first_month = int(min(months))

    # Produção
    prod = (ytg[(ytg["ano"]==year) & (ytg["mes"].isin(months))]
            .groupby(["Família Comercial","ano","mes"], as_index=False)["producao"].sum())

    # Estoque inicial agregado
    if df_estoque_inicial_agg is None:
        ini_agg, _aud = load_estoque_inicial_audit()
    else:
        ini_agg = df_estoque_inicial_agg.copy()

    if ini_agg.empty:
        ini_map = pd.DataFrame({
            "Família Comercial": fams,
            "ano": year, "mes": first_month, "estoque_inicial": 0.0
        })
    else:
        ini_map = ini_agg.copy()
        ini_map["ano"] = year; ini_map["mes"] = first_month
        missing = set(fams) - set(ini_map["Família Comercial"].astype(str))
        if missing:
            ini_map = pd.concat([
                ini_map,
                pd.DataFrame({"Família Comercial": sorted(list(missing)),
                              "ano": year, "mes": first_month, "estoque_inicial": 0.0})
            ], ignore_index=True)
    ini_map["estoque_inicial"] = _coerce_float(ini_map["estoque_inicial"])

    # Vendas — GATEWAY da DRE
    vend, _source = get_vendas_uc_long(ano=year, meses=months, prefer_ui=prefer_ui_for_sales)
    if not vend.empty:
        vend = vend[vend["mes"].isin(months)]

    # Universo
    base = pd.MultiIndex.from_product([fams, [year], months],
                                      names=["Família Comercial","ano","mes"]).to_frame(index=False)
    out = base.merge(ini_map, on=["Família Comercial","ano","mes"], how="left")
    out = out.merge(prod, on=["Família Comercial","ano","mes"], how="left")
    out = out.merge(vend, on=["Família Comercial","ano","mes"], how="left")

    out = _ensure_cols(out, {"estoque_inicial": 0.0, "producao": 0.0, "vendas": 0.0})
    out["estoque_inicial"] = _coerce_float(out["estoque_inicial"])
    out["producao"]        = _coerce_float(out["producao"])
    out["vendas"]          = _coerce_float(out["vendas"])

    # Carry (SEM clamp; pode negativo)
    out = out.sort_values(["Família Comercial","ano","mes"]).reset_index(drop=True)
    estoque_final = []
    last_by_fam: Dict[str, float] = {}
    for _, r in out.iterrows():
        fam = r["Família Comercial"]; m = int(r["mes"])
        ei = float(r["estoque_inicial"]) if m == first_month else last_by_fam.get(fam, 0.0)
        ef = ei + float(r["producao"]) - float(r["vendas"])
        estoque_final.append(ef)
        last_by_fam[fam] = ef

    out["estoque_final"] = estoque_final
    out["mes_nome"] = out["mes"].map(MONTH_MAP_PT)

    return out[[
        "Família Comercial","ano","mes","mes_nome",
        "estoque_inicial","producao","vendas","estoque_final"
    ]].copy()

def build_rolagem_estoque_wide(
    prefer_ui_for_sales: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retorna (LONG, WIDE):
      - LONG: ['Família Comercial','ano','mes','mes_nome','estoque_inicial','producao','vendas','estoque_final']
      - WIDE: 'Inicial' + blocos '<Mes> Produção', '<Mes> Vendas', '<Mes> Estoque Final'
    """
    roll = build_rolagem_estoque_base(prefer_ui_for_sales=prefer_ui_for_sales)
    if roll.empty:
        return roll, pd.DataFrame()

    months = sorted(roll["mes"].dropna().astype(int).unique().tolist())
    month_names = {m: MONTH_MAP_PT.get(int(m), str(m)) for m in months}
    first_month = int(min(months))

    inicial = (roll[roll["mes"] == first_month]
               .groupby("Família Comercial", as_index=False)["estoque_inicial"]
               .sum()
               .rename(columns={"estoque_inicial":"Inicial"}))

    pivot = roll.pivot_table(
        index="Família Comercial",
        columns="mes",
        values={"producao":"sum","vendas":"sum","estoque_final":"sum"},
        aggfunc="sum",
        fill_value=0.0,
    )

    ordered_cols, flat_names = [], []
    for m in months:
        mname = month_names[m]
        ordered_cols.extend([("producao", m), ("vendas", m), ("estoque_final", m)])
        flat_names.extend([f"{mname} Produção", f"{mname} Vendas", f"{mname} Estoque Final"])

    pivot = pivot.reindex(columns=pd.MultiIndex.from_tuples(ordered_cols))
    pivot.columns = flat_names
    pivot = pivot.reset_index()

    wide = inicial.merge(pivot, on="Família Comercial", how="right").fillna(0.0)
    num_cols = [c for c in wide.columns if c != "Família Comercial"]
    for c in num_cols:
        wide[c] = pd.to_numeric(wide[c], errors="coerce").fillna(0.0)

    roll = roll.sort_values(["Família Comercial","ano","mes"]).reset_index(drop=True)
    return roll, wide

# ------------------ (Opcional) Shim p/ compatibilidade legado ---------------- #
def build_rolagem_estoque(**kwargs):
    """
    Compat: retorna (roll_long, None) para código legado que esperava
    'build_rolagem_estoque'. Prefira usar build_rolagem_estoque_wide().
    """
    roll = build_rolagem_estoque_base(**kwargs)
    return roll, None
