# pages/05_Dashboard.py
# --------------------------------------------------------------------------------------
# Dashboard P&L (BP vs Projeção) — SEM SIMULADOR E SEM CONTROLES LATERAIS
# Projeção = YTD Realizado + YTG a partir dos Volumes (UI override sobre RES → RE)
# --------------------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ML / Otimização
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pulp  # para otimização de mix (LP)

# --- Motor já existente no projeto
from core.calculator import (
    build_simulado_pivot,
    MONTHS_PT,
    _load_base_calculos,
    _param_lookup,
)

st.set_page_config(page_title="Dashboard P&L — CFO", page_icon="📈", layout="wide")

# Insere logo Leao
from core.ui_branding import inject_sidebar_logo_bottom
inject_sidebar_logo_bottom("models/logo.jpeg", max_width_px=180)


# --------------------------------------------------------------------------------------
# Caminhos
# --------------------------------------------------------------------------------------
PARQUET_PATH = Path("data/current.parquet")
BASE_CALC_PATH = Path("data/premissas_pnl/base_calculos.xlsx")

# --- Arquivo RES
RES_WORKING_PATH = Path("data/res/res_working.parquet")
MONTHS_LOWER = {name.lower(): (m, name) for m, name in MONTHS_PT.items()}

VOLUME_ID = "volume_uc"
CONV_PCT = 0.256
DME_PCT_DEFAULT = 0.102

# --------------------------------------------------------------------------------------
# Utilitários de carga / cenário / pivot
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for c in ("ano", "mes"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df["indicador_id"] = df["indicador_id"].astype(str)
    df["cenario"] = df["cenario"].astype(str).str.strip()
    if "valor" in df.columns:
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce").fillna(0.0)
    fam_cols = [c for c in df.columns if str(c).lower().startswith("fam")]
    if fam_cols:
        df.rename(columns={fam_cols[0]: "Família Comercial"}, inplace=True)
    return df

def _is_realizado(name: str) -> bool:
    return isinstance(name, str) and ("realiz" in name.lower())

def _is_bp(name: str) -> bool:
    return isinstance(name, str) and name.lower().replace(" ", "").startswith("bp")

def list_years(df: pd.DataFrame) -> List[int]:
    return sorted([int(x) for x in df["ano"].dropna().unique()])

def find_bp_scenario(df: pd.DataFrame, year: int) -> Optional[str]:
    cand = df.loc[df["ano"] == year, "cenario"].dropna().unique().tolist()
    for s in cand:
        if _is_bp(str(s)):
            return s
    return None

def find_realizado_scenario(df: pd.DataFrame, year: int) -> Optional[str]:
    cand = df.loc[df["ano"] == year, "cenario"].dropna().unique().tolist()
    for s in cand:
        if _is_realizado(str(s)):
            return s
    return None

def cutoff_from_realizado_auto(df: pd.DataFrame, year: int, re_scen: Optional[str]) -> int:
    if re_scen is None:
        return 0
    sub = df[(df["ano"] == year) & (df["cenario"] == re_scen) & (df["indicador_id"] == VOLUME_ID)]
    if sub.empty:
        return 0
    g = sub.groupby("mes")['valor'].sum()
    g = g[g != 0]
    return int(g.index.max()) if not g.empty else 0

def pivot_scenario(df: pd.DataFrame, year: int, scenario: str) -> pd.DataFrame:
    month_cols = [MONTHS_PT[m] for m in range(1, 13)]
    base = df[(df["ano"] == year) & (df["cenario"] == scenario)].copy()
    if base.empty:
        return pd.DataFrame(
            [{"indicador_id": "empty", "Indicador": "Empty", **{mc: 0 for mc in month_cols}, "Total Ano": 0}]
        )
    agg = (
        base.groupby(["indicador_id", "mes"], as_index=False)["valor"].sum()
           .pivot(index="indicador_id", columns="mes", values="valor")
           .reindex(columns=range(1, 13))
           .fillna(0.0)
    )
    rows = []
    for rid in agg.index:
        vals = {MONTHS_PT[m]: float(agg.loc[rid, m] if m in agg.columns else 0.0) for m in range(1, 13)}
        rows.append({
            "indicador_id": rid,
            "Indicador": rid,
            **{k: int(np.rint(v)) for k, v in vals.items()},
            "Total Ano": int(np.rint(sum(vals.values()))),
        })
    out = pd.DataFrame(rows)
    return out[["indicador_id", "Indicador", *month_cols, "Total Ano"]]

def sum_row(piv: pd.DataFrame, rid: str) -> int:
    row = piv.loc[piv["indicador_id"] == rid]
    return int(row["Total Ano"].values[0]) if not row.empty else 0

# --------------------------------------------------------------------------------------
# RES — leitura robusta (independente da tela de Volumes)
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_res_working() -> Optional[pd.DataFrame]:
    if not RES_WORKING_PATH.exists():
        return None
    try:
        wrk = pd.read_parquet(RES_WORKING_PATH)
    except Exception:
        return None
    if wrk is None or wrk.empty:
        return None
    low = {str(c).lower(): c for c in wrk.columns}
    ano_col = low.get("ano") or next((c for c in wrk.columns if str(c).lower() in {"ano", "year"}), None)
    mes_col = low.get("mes") or low.get("mês") or next((c for c in wrk.columns if str(c).lower() in {"mes", "mês", "month"}), None)
    fam_col = low.get("família comercial") or low.get("familia comercial") or next((c for c in wrk.columns if str(c).lower().startswith("fam")), None)
    vol_col = next((c for c in wrk.columns if "volume" in str(c).lower()
                    or str(c).lower() in {"volume_uc", "vol", "qtd", "quantidade", "uc", "valor"}), None)
    if not all([ano_col, mes_col, fam_col, vol_col]):
        return None
    wrk = wrk.rename(columns={fam_col: "Família Comercial"})
    wrk["ano"] = pd.to_numeric(wrk[ano_col], errors="coerce").astype("Int64")
    wrk["mes"] = pd.to_numeric(wrk[mes_col], errors="coerce").astype("Int64")
    wrk["volume"] = pd.to_numeric(wrk[vol_col], errors="coerce").fillna(0.0)
    return wrk[["ano", "mes", "Família Comercial", "volume"]]

# --------------------------------------------------------------------------------------
# (NOVO) Projetado (YTG) — Totais mensais para Insights: UI -> RES -> RE
# NÃO exige coluna de família no RES e independe de abrir a tela de Volumes.
# --------------------------------------------------------------------------------------
def _month_totals_from_ui(year: int) -> Optional[Dict[int, int]]:
    # 1) volumes_wide
    vw = st.session_state.get("volumes_wide")
    if isinstance(vw, dict) and year in vw and isinstance(vw[year], pd.DataFrame):
        dfw = vw[year]
        col_map = {}
        for c in dfw.columns:
            lc = str(c).strip().lower()
            if lc in MONTHS_LOWER:
                col_map[MONTHS_LOWER[lc][0]] = c
        if col_map:
            out = {}
            for m in range(1, 13):
                col = col_map.get(m)
                out[m] = 0 if col is None else int(pd.to_numeric(dfw[col], errors="coerce").fillna(0.0).sum())
            return out
    # 2) volumes_edit (long)
    ve = st.session_state.get("volumes_edit")
    if isinstance(ve, pd.DataFrame) and {"ano", "mes", "volume"}.issubset(ve.columns):
        sub = ve[ve["ano"] == year]
        if not sub.empty:
            g = pd.to_numeric(sub["volume"], errors="coerce").fillna(0.0).groupby(sub["mes"]).sum()
            return {int(m): int(g.get(m, 0.0)) for m in range(1, 13)}
    return None

def _month_totals_from_res(year: int) -> Optional[Dict[int, int]]:
    if not RES_WORKING_PATH.exists():
        return None
    try:
        wrk = pd.read_parquet(RES_WORKING_PATH)
    except Exception:
        return None
    if wrk is None or wrk.empty:
        return None
    lower = {str(c).lower(): c for c in wrk.columns}
    ano_col = lower.get("ano") or next((c for c in wrk.columns if str(c).lower() in {"ano", "year"}), None)
    mes_col = lower.get("mes") or lower.get("mês") or next((c for c in wrk.columns if str(c).lower() in {"mes", "mês", "month"}), None)
    vol_col = next((c for c in wrk.columns if "volume" in str(c).lower()
                    or str(c).lower() in {"volume_uc", "vol", "qtd", "quantidade", "uc", "valor"}), None)
    if not all([ano_col, mes_col, vol_col]):
        return None
    sub = wrk[wrk[ano_col] == year]
    if sub.empty:
        return None
    g = pd.to_numeric(sub[vol_col], errors="coerce").fillna(0.0).groupby(sub[mes_col]).sum()
    return {int(m): int(g.get(m, 0.0)) for m in range(1, 13)}

def _month_totals_from_re(df: pd.DataFrame, year: int) -> Dict[int, int]:
    scen = None
    cand = df.loc[df["ano"] == year, "cenario"].dropna().unique().tolist()
    for s in cand:
        if str(s).strip().lower().startswith("re"):
            scen = s
            break
    sub = df[(df["ano"] == year) & (df["cenario"] == scen) & (df["indicador_id"] == VOLUME_ID)]
    g = sub.groupby("mes")["valor"].sum()
    return {int(m): int(g.get(m, 0.0)) for m in range(1, 13)}

def get_sim_month_totals(df: pd.DataFrame, year: int) -> Dict[int, int]:
    """Totais mensais de volume para YTG (Insights): UI -> RES -> RE."""
    for getter in (_month_totals_from_ui, _month_totals_from_res):
        try:
            out = getter(year)
            if out and sum(out.values()) > 0:
                return out
        except Exception:
            pass
    return _month_totals_from_re(df, year)

# --------------------------------------------------------------------------------------
# Compositor de baseline YTG (RES → RE) com override da UI (apenas onde difere)
# Entrega SEMPRE um DataFrame no formato volumes_edit (long) para o core.
# --------------------------------------------------------------------------------------
def _normalize_ui_long(df_ui: Optional[pd.DataFrame], year: int) -> pd.DataFrame:
    if not isinstance(df_ui, pd.DataFrame) or df_ui.empty:
        return pd.DataFrame(columns=["ano", "mes", "Família Comercial", "volume"])
    df = df_ui.copy()
    low = {str(c).lower(): c for c in df.columns}
    ano_col = low.get("ano") or "ano"
    mes_col = low.get("mes") or "mes"
    vol_col = low.get("volume") or low.get("volume_uc") or "volume"
    fam_col = low.get("família comercial") or low.get("familia comercial")
    if fam_col is None:
        fam_col = next((c for c in df.columns if str(c).lower().startswith("fam")), None)
    if fam_col is None:
        df["_fam_tmp_"] = "TOTAL"
        fam_col = "_fam_tmp_"
    df["ano"] = pd.to_numeric(df[ano_col], errors="coerce").fillna(year).astype(int)
    df["mes"] = pd.to_numeric(df[mes_col], errors="coerce").astype(int)
    df["volume"] = pd.to_numeric(df[vol_col], errors="coerce").fillna(0.0)
    df.rename(columns={fam_col: "Família Comercial"}, inplace=True)
    return df[["ano", "mes", "Família Comercial", "volume"]]

def _re_as_long(df_all: pd.DataFrame, year: int, cutoff: int) -> pd.DataFrame:
    re_scen = find_realizado_scenario(df_all, year)
    if re_scen is None:
        return pd.DataFrame(columns=["ano", "mes", "Família Comercial", "volume"])
    base = df_all[(df_all["ano"] == year) & (df_all["cenario"] == re_scen) & (df_all["indicador_id"] == VOLUME_ID)]
    if base.empty:
        return pd.DataFrame(columns=["ano", "mes", "Família Comercial", "volume"])
    if "Família Comercial" not in base.columns:
        base = base.copy()
        base["Família Comercial"] = "TOTAL"
    ytg_months = list(range(cutoff + 1, 13))
    rows = []
    fams = base["Família Comercial"].astype(str).unique().tolist()
    for f in fams:
        for m in ytg_months:
            rows.append({"ano": year, "mes": m, "Família Comercial": f, "volume": 0.0})
    return pd.DataFrame(rows)

def compose_plan_volumes_long(
    *, df_all: pd.DataFrame, year: int, cutoff: int, ui_long: Optional[pd.DataFrame]
) -> Tuple[pd.DataFrame, str]:
    res_df = load_res_working()
    ytg = list(range(cutoff + 1, 13))
    if res_df is not None and not res_df.empty and int(year) in res_df["ano"].dropna().astype(int).unique():
        base = res_df[(res_df["ano"] == year) & (res_df["mes"].isin(ytg))][["ano", "mes", "Família Comercial", "volume"]].copy()
        fonte = "RES"
    else:
        base = _re_as_long(df_all, year, cutoff)
        fonte = "RE"
    ui_norm = _normalize_ui_long(ui_long, year)
    if not ui_norm.empty:
        ui_sub = ui_norm[(ui_norm["ano"] == year) & (ui_norm["mes"].isin(ytg))].copy()
        if not ui_sub.empty:
            ui_grp = (ui_sub.groupby(["ano", "mes", "Família Comercial"], as_index=False)["volume"].sum())
            if base.empty:
                base = ui_grp.copy()
                fonte = "UI>RE"
            else:
                for r in ui_grp.itertuples(index=False):
                    fam = getattr(r, "Família Comercial", None)
                    if fam is None:
                        fam = getattr(r, "Familia Comercial", None)
                    if fam is None:
                        fam = r[2] if isinstance(r[2], str) else str(r[2])
                    fam = str(fam)
                    k = (int(r.ano), int(r.mes), fam)
                    mask = (base["ano"].eq(k[0]) & base["mes"].eq(k[1]) & base["Família Comercial"].astype(str).eq(k[2]))
                    if mask.any():
                        base.loc[mask, "volume"] = float(r.volume)
                    else:
                        base = pd.concat([base, pd.DataFrame([{"ano": k[0], "mes": k[1], "Família Comercial": k[2], "volume": float(r.volume)}])], ignore_index=True)
                fonte = "UI>RES" if fonte == "RES" else "UI>RE"
    if base.empty:
        base = pd.DataFrame(columns=["ano", "mes", "Família Comercial", "volume"])
    base["ano"] = year
    base["mes"] = base["mes"].astype(int)
    base["volume"] = pd.to_numeric(base["volume"], errors="coerce").fillna(0.0)
    if "Família Comercial" not in base.columns:
        base["Família Comercial"] = "TOTAL"
    base = base[(base["ano"] == year) & (base["mes"] > cutoff)]
    return base.reset_index(drop=True), fonte

def month_totals_from_plan(plan_long: pd.DataFrame, year: int, cutoff: int) -> Dict[int, int]:
    if plan_long is None or plan_long.empty:
        return {m: 0 for m in range(1, 13)}
    g = plan_long.groupby("mes")["volume"].sum()
    out = {int(m): int(round(g.get(m, 0.0))) for m in range(1, 13)}
    for m in range(1, cutoff + 1):
        out[m] = 0
    return out

# --------------------------------------------------------------------------------------
# Helpers de cálculo unitário (parâmetros)
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_params() -> pd.DataFrame:
    return _load_base_calculos(BASE_CALC_PATH)

def _rb_uc(params: pd.DataFrame, fam: str, m: int) -> float:
    return float(_param_lookup(params, fam, m, "Receita Bruta", "Valor") or 0.0)

def _ins_uc(params: pd.DataFrame, fam: str, m: int) -> float:
    return float(_param_lookup(params, fam, m, "Insumos", "Valor") or 0.0)

def _toll_uc(params: pd.DataFrame, fam: str, m: int) -> float:
    return float(_param_lookup(params, fam, m, "Custos Toll Packer", "Valor") or 0.0)

def _frete_uc(params: pd.DataFrame, fam: str, m: int) -> float:
    return float(_param_lookup(params, fam, m, "Fretes T1", "Valor") or 0.0)

def rl_uc_from_rb(rb_uc: float) -> float:
    return (1.0 - CONV_PCT) * rb_uc

# ============================
# Helpers de ML
# ============================
def _impute_numeric_frame(X: pd.DataFrame, ref_means: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series]:
    Xc = X.copy().replace([np.inf, -np.inf], np.nan)
    means = Xc.mean(numeric_only=True) if ref_means is None else ref_means
    Xc = Xc.fillna(means).fillna(0.0)
    return Xc, means

def _safe_gbr_train_pred(X: pd.DataFrame, y: np.ndarray, Xf: pd.DataFrame) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    mask_y = np.isfinite(y)
    X = X.loc[mask_y].copy(); y = y[mask_y]
    X, means = _impute_numeric_frame(X, ref_means=None)
    if len(X) < 3:
        model = LinearRegression()
    elif len(X) < 6:
        model = GradientBoostingRegressor(random_state=42, max_depth=2, n_estimators=200, learning_rate=0.05)
    else:
        model = GradientBoostingRegressor(random_state=42)
    model.fit(X, y)
    Xf, _ = _impute_numeric_frame(Xf, ref_means=means)
    yh = model.predict(Xf)
    return np.maximum(yh, 0.0)

def _global_monthly_ml_predict(df: pd.DataFrame, year: int, indicator_id: str, months: List[int]) -> Dict[int, float]:
    mask = (df["indicador_id"].eq(indicator_id) & df["ano"].lt(year) & df["cenario"].apply(lambda s: _is_realizado(str(s))))
    hist = df[mask].groupby(["ano", "mes"])["valor"].sum().reset_index()
    if hist.empty or not months:
        return {m: 0.0 for m in months}
    hist = hist.sort_values(["ano", "mes"]).reset_index(drop=True)
    def lag12_lookup(a, m):
        v = hist[(hist["ano"] == a - 1) & (hist["mes"] == m)]["valor"]
        return float(v.values[0]) if len(v) else np.nan
    hist["lag12"] = [lag12_lookup(int(r.ano), int(r.mes)) for r in hist.itertuples()]
    hist["lag12"] = hist.groupby("mes")["lag12"].transform(lambda s: s.fillna(s.mean()))
    hist["lag12"] = hist["lag12"].fillna(hist["valor"].mean())

    X = pd.DataFrame({
        "mes": hist["mes"].astype(int),
        "sin": np.sin(2 * np.pi * hist["mes"].astype(float) / 12.0),
        "cos": np.cos(2 * np.pi * hist["mes"].astype(float) / 12.0),
        "ano": hist["ano"].astype(int),
        "lag12": hist["lag12"].astype(float),
    })
    y = hist["valor"].astype(float).values
    Xf = pd.DataFrame({
        "mes": months,
        "sin": np.sin(2 * np.pi * np.array(months) / 12.0),
        "cos": np.cos(2 * np.pi * np.array(months) / 12.0),
        "ano": [year] * len(months),
        "lag12": [lag12_lookup(year, m) if not np.isnan(lag12_lookup(year, m)) else hist[hist["mes"]==m]["valor"].mean() for m in months],
    })
    yh = _safe_gbr_train_pred(X, y, Xf)
    return {int(m): float(max(0.0, v)) for m, v in zip(months, yh)}

def _ml_predict_ytg_by_family_smart(df: pd.DataFrame, year: int, ytg_months: List[int], fam_col: str = "Família Comercial") -> Dict[Tuple[str, int], float]:
    if not ytg_months:
        return {}
    global_ml = _global_monthly_ml_predict(df, year, VOLUME_ID, ytg_months)
    last_year = year - 1
    mask_last = (df["indicador_id"].eq(VOLUME_ID) & df["ano"].eq(last_year) & df["cenario"].apply(lambda s: _is_realizado(str(s))))
    by_fam_last = df[mask_last].groupby([fam_col, "mes"])["valor"].sum().reset_index()
    total_last = df[mask_last].groupby("mes")["valor"].sum().rename("tot")
    by_fam_last = by_fam_last.merge(total_last, on="mes", how="left")
    by_fam_last["share"] = np.where(by_fam_last["tot"] > 0, by_fam_last["valor"] / by_fam_last["tot"], 0.0)
    shares = {(str(r[fam_col]), int(r["mes"])): float(r["share"]) for _, r in by_fam_last.iterrows()}

    mask_hist = (df["indicador_id"].eq(VOLUME_ID) & df["ano"].lt(year) & df["cenario"].apply(lambda s: _is_realizado(str(s))))
    hist = df[mask_hist].copy()
    if fam_col not in hist.columns or hist.empty:
        return {("TOTAL", m): v for m, v in global_ml.items()}

    preds: Dict[Tuple[str, int], float] = {}
    for fam, g in hist.groupby(fam_col):
        g = g[["ano", "mes", "valor"]].dropna()
        if g.empty:
            for m in ytg_months:
                share = shares.get((str(fam), m), 0.0)
                preds[(str(fam), m)] = float(max(0.0, share * global_ml.get(m, 0.0)))
            continue
        def fam_lag12(a, m):
            v = g[(g["ano"] == a - 1) & (g["mes"] == m)]["valor"]
            return float(v.values[0]) if len(v) else np.nan
        g = g.sort_values(["ano", "mes"])
        g["lag12"] = [fam_lag12(int(r.ano), int(r.mes)) for r in g.itertuples()]
        g["lag12"] = g.groupby("mes")["lag12"].transform(lambda s: s.fillna(s.mean()))
        g["lag12"] = g["lag12"].fillna(g["valor"].mean())

        X = pd.DataFrame({
            "mes": g["mes"].astype(int),
            "sin": np.sin(2 * np.pi * g["mes"].astype(float) / 12.0),
            "cos": np.cos(2 * np.pi * g["mes"].astype(float) / 12.0),
            "ano": g["ano"].astype(int),
            "lag12": g["lag12"].astype(float),
        })
        y = g["valor"].astype(float).values
        if len(X) < 6:
            mean_by_month = g.groupby("mes")["valor"].mean().to_dict()
            for m in ytg_months:
                base = float(mean_by_month.get(m, np.nan))
                if np.isnan(base):
                    share = shares.get((str(fam), m), 0.0)
                    base = float(share * global_ml.get(m, 0.0))
                preds[(str(fam), m)] = max(0.0, base)
            continue
        Xf = pd.DataFrame({
            "mes": ytg_months,
            "sin": np.sin(2 * np.pi * np.array(ytg_months) / 12.0),
            "cos": np.cos(2 * np.pi * np.array(ytg_months) / 12.0),
            "ano": [year] * len(ytg_months),
            "lag12": [fam_lag12(year, m) if not np.isnan(fam_lag12(year, m)) else g[g["mes"]==m]["valor"].mean() for m in ytg_months],
        })
        yh = _safe_gbr_train_pred(X, y, Xf)
        for m, vhat in zip(ytg_months, yh):
            share = shares.get((str(fam), m), 0.0)
            fallback = share * global_ml.get(m, 0.0)
            v_adj = max(0.1 * fallback, float(vhat), 0.0)
            preds[(str(fam), int(m))] = v_adj
    return preds

# --------------------------------------------------------------------------------------
# INÍCIO — Carregamento de dados
# --------------------------------------------------------------------------------------
df = load_parquet(PARQUET_PATH)
years = list_years(df)
year = years[-1] if years else pd.Timestamp.today().year

bp_scen = find_bp_scenario(df, year)
re_scen = find_realizado_scenario(df, year)

if bp_scen is None or re_scen is None:
    st.error("Não encontrei cenários BP e/ou Realizado no parquet.")
    st.stop()

cutoff = cutoff_from_realizado_auto(df, year, re_scen)
dme_pct = DME_PCT_DEFAULT

st.markdown("## Dashboard P&L — **BP vs Projeção**")

# --------------------------------------------------------------------------------------
# Projeção (YTD RE + YTG com baseline RES→RE e override UI)
# --------------------------------------------------------------------------------------
piv_bp = pivot_scenario(df, year, bp_scen)
piv_real = pivot_scenario(df, year, re_scen)

volumes_ui_long = st.session_state.get("volumes_edit")

# 1) Monta baseline planejado (YTG) = RES→RE com override da UI
plan_long, fonte_plan = compose_plan_volumes_long(df_all=df, year=year, cutoff=cutoff, ui_long=volumes_ui_long)

# 2) Totais mensais para o core (somente YTG; YTD=0)
ui_month_totals = month_totals_from_plan(plan_long, year, cutoff)

# 3) Sempre enviamos como "ui" para o core (mesmo quando veio do RES)
volume_mode_core = "ui" if not plan_long.empty else "re"

piv_proj = build_simulado_pivot(
    df=df,
    piv_real=piv_real,
    year=year,
    cutoff=cutoff,
    base_calc_path=BASE_CALC_PATH,
    volumes_edit=plan_long if not plan_long.empty else None,
    volumes_res=None,                # não usamos mais
    volume_mode=volume_mode_core,    # "ui" (com plan_long) ou "re"
    dme_pct=float(dme_pct),
    ui_month_totals=ui_month_totals,
    conv_source="excel",
)

st.caption(f"Fonte YTG: **{fonte_plan}** (enviado ao core como `volume_mode=ui`).")

# KPI helpers
def total_full_year(piv: pd.DataFrame, rid: str) -> float:
    return float(sum_row(piv, rid))

rl_proj, rl_bp = total_full_year(piv_proj, "receita_liquida"), total_full_year(piv_bp, "receita_liquida")
lb_proj, lb_bp = total_full_year(piv_proj, "margem_bruta"), total_full_year(piv_bp, "margem_bruta")
ro_proj, ro_bp = total_full_year(piv_proj, "resultado_operacional"), total_full_year(piv_bp, "resultado_operacional")

# --------------------------------------------------------------------------------------
# ABAS
# --------------------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Painel do CFO - Diagnóstico Estratégico",
    "Análise de Portfólio e Sensibilidade",
    "Simulação What-If & Detalhes",
    "Insights Preditivos & Otimização (ML)",
])

# ============================
# TAB 1 — Painel do CFO
# ============================
with tab1:
    c1, c2, c3 = st.columns(3)
    def kpi(col, titulo, val_proj, val_bp):
        diff = val_proj - val_bp
        pct = (diff / val_bp * 100.0) if val_bp != 0 else 0.0
        with col:
            st.metric(titulo, f"{val_proj:,.0f}".replace(",", "."), f"{diff:,.0f} ({pct:+.1f}%)".replace(",", "."))
    kpi(c1, "Receita Líquida (Proj)", rl_proj, rl_bp)
    kpi(c2, "Lucro Bruto (Proj)",     lb_proj, lb_bp)
    kpi(c3, "Resultado Operacional (Proj)", ro_proj, ro_bp)

    st.markdown("---")

    st.subheader("Bridge Anual do Resultado Operacional — BP25 vs Projeção (Jan..Dez + FY)")
    months_labels = [MONTHS_PT[m] for m in range(1, 13)]
    def row_series(piv: pd.DataFrame, rid: str) -> List[float]:
        if rid not in piv["indicador_id"].values:
            return [0.0]*12
        row = piv.loc[piv["indicador_id"] == rid, months_labels]
        return [float(row[c].values[0]) if not row.empty else 0.0 for c in months_labels]
    ro_bp_months   = row_series(piv_bp, "resultado_operacional")
    ro_proj_months = row_series(piv_proj, "resultado_operacional")
    deltas = [proj - bp for bp, proj in zip(ro_bp_months, ro_proj_months)]
    wf_x = ["RO BP (Ano)"] + months_labels + ["RO Projeção (Ano)"]
    wf_y = [ro_bp] + deltas + [ro_proj]
    measures = ["absolute"] + ["relative"]*12 + ["total"]
    fig_wf_monthly = go.Figure(go.Waterfall(x=wf_x, y=wf_y, measure=measures, connector={"line": {"width": 1}}))
    fig_wf_monthly.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=10), xaxis_title=None, yaxis_title="R$")
    st.plotly_chart(fig_wf_monthly, use_container_width=True)

    st.markdown("---")

    st.subheader("Comparativo Histórico — YoY por mês (Ano corrente = YTD Realizado + YTG Projetado)")
    years_all = list_years(df)
    def mixed_series_current_year(rid: str) -> List[float]:
        real = row_series(piv_real, rid)
        proj = row_series(piv_proj, rid)
        out = []
        for i in range(12):
            m = i + 1
            out.append(real[i] if m <= cutoff else proj[i])
        return out
    hist_rows = []
    for y in years_all:
        scen_r = find_realizado_scenario(df, y)
        if scen_r is None:
            continue
        piv_r = pivot_scenario(df, y, scen_r)
        if y == year:
            for rid, label in [("volume_uc", "Volume (UC)"), ("receita_liquida", "Receita Líquida"), ("resultado_operacional", "Resultado Operacional")]:
                vals = mixed_series_current_year(rid)
                for mi, val in enumerate(vals, start=1):
                    hist_rows.append({"Ano": y, "Mês": MONTHS_PT[mi], "Indicador": label, "Valor": float(val)})
        else:
            def series_rid(_rid: str) -> List[float]:
                if _rid not in piv_r["indicador_id"].values:
                    return [0.0]*12
                row = piv_r.loc[piv_r["indicador_id"] == _rid, months_labels]
                return [float(row[c].values[0]) if not row.empty else 0.0 for c in months_labels]
            for rid, label in [("volume_uc", "Volume (UC)"), ("receita_liquida", "Receita Líquida"), ("resultado_operacional", "Resultado Operacional")]:
                vals = series_rid(rid)
                for mi, val in enumerate(vals, start=1):
                    hist_rows.append({"Ano": y, "Mês": MONTHS_PT[mi], "Indicador": label, "Valor": float(val)})
    if hist_rows:
        df_hist = pd.DataFrame(hist_rows)
        df_plot = df_hist[df_hist["Indicador"] == "Volume (UC)"]
        fig_vol = px.line(df_plot, x="Mês", y="Valor", color="Ano", markers=True, title="Volume (UC) — YoY por mês (Ano corrente com YTG projetado)")
        fig_vol.update_layout(height=320, margin=dict(l=10, r=20, t=30, b=10))
        st.plotly_chart(fig_vol, use_container_width=True)
        df_plot = df_hist[df_hist["Indicador"] == "Receita Líquida"]
        fig_rl = px.line(df_plot, x="Mês", y="Valor", color="Ano", markers=True, title="Receita Líquida — YoY por mês (Ano corrente com YTG projetado)")
        fig_rl.update_layout(height=320, margin=dict(l=10, r=20, t=30, b=10))
        st.plotly_chart(fig_rl, use_container_width=True)
        df_plot = df_hist[df_hist["Indicador"] == "Resultado Operacional"]
        fig_ro = px.line(df_plot, x="Mês", y="Valor", color="Ano", markers=True, title="Resultado Operacional — YoY por mês (Ano corrente com YTG projetado)")
        fig_ro.update_layout(height=320, margin=dict(l=10, r=20, t=30, b=10))
        st.plotly_chart(fig_ro, use_container_width=True)
    else:
        st.info("Não foram encontrados anos com cenário Realizado para o comparativo.")

# ============================
# TAB 2 — Análise de Portfólio e Sensibilidade
# ============================
with tab2:
    fam_col = "Família Comercial"
    bp_ytg = df[(df["ano"] == year) & (df["cenario"] == bp_scen) & (df["indicador_id"] == VOLUME_ID) & (df["mes"] > cutoff)]

    st.subheader("Sensibilidade do RO (±10% no volume) — Rankings")
    if not bp_ytg.empty and fam_col in bp_ytg.columns:
        v_bp_fam = bp_ytg.groupby(fam_col)["valor"].sum().to_dict()
        params = load_params()
        rows = []
        for fam, vtot in v_bp_fam.items():
            fam = str(fam)
            if cutoff < 12:
                rbuc = np.mean([_rb_uc(params, fam, m) for m in range(cutoff + 1, 13)])
                var_uc = np.mean([_ins_uc(params, fam, m) + _toll_uc(params, fam, m) + _frete_uc(params, fam, m) for m in range(cutoff + 1, 13)])
            else:
                rbuc, var_uc = 0.0, 0.0
            rluc = rl_uc_from_rb(rbuc)
            slope = (rluc - var_uc) - (DME_PCT_DEFAULT * rluc)
            impacto_10 = slope * (0.10 * float(vtot))
            rows.append({"Familia": fam, "Impacto (+10%)": impacto_10})
        df_sens = pd.DataFrame(rows)
        df_sens["|Impacto|"] = df_sens["Impacto (+10%)"].abs()

        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**Top 10 que mais afetam (por magnitude)**")
            top_more = df_sens.sort_values("|Impacto|", ascending=False).head(10)
            fig_top_more = px.bar(top_more, x="Impacto (+10%)", y="Familia", orientation="h", title=None)
            fig_top_more.update_layout(height=420, margin=dict(l=10, r=20, t=10, b=10))
            st.plotly_chart(fig_top_more, use_container_width=True)
        with col_b:
            st.write("**Top 10 que menos afetam (por magnitude)**")
            top_less = df_sens.sort_values("|Impacto|", ascending=True).head(10)
            fig_top_less = px.bar(top_less, x="Impacto (+10%)", y="Familia", orientation="h", title=None)
            fig_top_less.update_layout(height=420, margin=dict(l=10, r=20, t=10, b=10))
            st.plotly_chart(fig_top_less, use_container_width=True)
    else:
        st.info("Sem dados de BP YTG por família para sensibilidade.")

    st.markdown("---")

    st.subheader("Portfólio de Crescimento — Top 10 categorias que mais cresceram (histórico anual)")
    hist_years = [y for y in years if y < year]
    if len(hist_years) >= 2 and fam_col in df.columns:
        real_mask = df["cenario"].apply(lambda s: _is_realizado(str(s)))
        vol_hist = (df[real_mask & df["indicador_id"].eq(VOLUME_ID) & df["ano"].isin(hist_years)]
                    .groupby(["ano", fam_col])["valor"].sum().reset_index())
        first_year, last_year = min(hist_years), max(hist_years)
        base = vol_hist[vol_hist["ano"] == first_year].set_index(fam_col)["valor"]
        last = vol_hist[vol_hist["ano"] == last_year].set_index(fam_col)["valor"]
        fams_all = sorted(set(base.index.astype(str)) | set(last.index.astype(str)))
        rows = []
        for fam in fams_all:
            v0 = float(base.get(fam, 0.0)); v1 = float(last.get(fam, 0.0))
            growth = (v1 - v0) / v0 * 100.0 if v0 > 0 else (100.0 if v1 > 0 else 0.0)
            rows.append({"Familia": str(fam), "Crescimento (%)": growth, "Valor Inicial": v0, "Valor Final": v1})
        rank_growth = pd.DataFrame(rows).sort_values("Crescimento (%)", ascending=False).head(10)
        fig_rank = px.bar(rank_growth, x="Crescimento (%)", y="Familia", orientation="h", title=f"Top 10 Crescimento — {first_year} → {last_year}")
        fig_rank.update_layout(height=420, margin=dict(l=10, r=20, t=10, b=10))
        st.plotly_chart(fig_rank, use_container_width=True)

        top_fams = set(rank_growth["Familia"].astype(str))
        vol_hist_top = vol_hist[vol_hist[fam_col].astype(str).isin(top_fams)].copy().rename(columns={fam_col: "Familia", "valor": "Volume"})
        fig_lines = px.line(vol_hist_top, x="ano", y="Volume", color="Familia", markers=True, title="Evolução Anual de Volume — Top 10 Crescimento (Realizado)")
        fig_lines.update_layout(height=420, margin=dict(l=10, r=20, t=10, b=10), xaxis=dict(dtick=1))
        st.plotly_chart(fig_lines, use_container_width=True)
    else:
        st.info("Histórico insuficiente (menos de 2 anos de Realizado) para calcular crescimento.")

    st.markdown("---")
    st.subheader("Concentração de Resultado: Quais famílias geram 80% do RO (YTG)?")
    df_delta = pd.DataFrame(columns=["Familia", "delta_ro"])  # placeholder
    if not df_delta.empty:
        df_par = df_delta[["Familia", "delta_ro"]].rename(columns={"delta_ro": "RO_Contrib"}).sort_values("RO_Contrib", ascending=False)
        df_par["% Acumulado"] = (df_par["RO_Contrib"].cumsum() / max(1e-9, df_par["RO_Contrib"].sum())) * 100.0
        fig_par = go.Figure()
        fig_par.add_bar(x=df_par["Familia"], y=df_par["RO_Contrib"], name="RO Contrib (Δ)")
        fig_par.add_scatter(x=df_par["Familia"], y=df_par["% Acumulado"], name="% Acum", yaxis="y2", mode="lines+markers")
        fig_par.update_layout(height=420, margin=dict(l=10, r=20, t=10, b=10),
                              yaxis=dict(title="RO Contrib (Δ)"),
                              yaxis2=dict(title="% Acum", overlaying="y", side="right", range=[0, 100]))
        st.plotly_chart(fig_par, use_container_width=True)
    else:
        st.info("Sem variação consolidada por família para Pareto.")

# ============================
# TAB 3 — What-If & Detalhes
# ============================
with tab3:
    st.subheader("Simulador de Ponto Ótimo — Curva de Resposta do RO")
    fam_col = "Família Comercial"
    bp_ytg = df[(df["ano"] == year) & (df["cenario"] == bp_scen) & (df["indicador_id"] == VOLUME_ID) & (df["mes"] > cutoff)]
    familles = sorted(bp_ytg["Família Comercial"].astype(str).unique()) if not bp_ytg.empty and "Família Comercial" in bp_ytg.columns else []
    fam_sel = st.selectbox("Família Alvo", familles, index=0 if familles else None)
    if fam_sel:
        xs = np.linspace(-0.5, 0.5, 41)
        params = load_params()
        rbuc = np.mean([_rb_uc(params, fam_sel, m) for m in range(cutoff + 1, 13)]) if cutoff < 12 else 0.0
        rluc = rl_uc_from_rb(rbuc)
        var_uc = np.mean([_ins_uc(params, fam_sel, m) + _toll_uc(params, fam_sel, m) + _frete_uc(params, fam_sel, m) for m in range(cutoff + 1, 13)]) if cutoff < 12 else 0.0
        slope = (rluc - var_uc) - (dme_pct * rluc)
        v_bp_base = float(bp_ytg.groupby("Família Comercial")["valor"].sum().get(fam_sel, 0.0)) if not bp_ytg.empty else 0.0
        ro_proj_base = ro_proj
        ys = []
        for dx in xs:
            dv = v_bp_base * dx
            d_ro = slope * dv
            ys.append(ro_proj_base + d_ro)
        fig_curve = px.line(x=xs * 100, y=ys, labels={"x": "%Δ Volume (Família)", "y": "RO Total (Proj)"})
        fig_curve.update_layout(height=400, margin=dict(l=10, r=20, t=20, b=10))
        st.plotly_chart(fig_curve, use_container_width=True)

    with st.expander("Análise de Sazonalidade da Rentabilidade (Heatmap MC/UC)", expanded=False):
        if cutoff < 12 and familles:
            params = load_params()
            heat = []
            for fam in familles:
                for m in range(1, 13):
                    rbuc = _rb_uc(params, fam, m)
                    rluc = rl_uc_from_rb(rbuc)
                    var_uc = _ins_uc(params, fam, m) + _toll_uc(params, fam, m) + _frete_uc(params, fam, m)
                    mc_uc = rluc - var_uc
                    heat.append({"Familia": fam, "Mes": MONTHS_PT[m], "MC/UC": mc_uc})
            df_heat = pd.DataFrame(heat)
            fig_h = px.density_heatmap(df_heat, x="Mes", y="Familia", z="MC/UC", histfunc="avg", color_continuous_scale="RdYlGn")
            fig_h.update_layout(height=540, margin=dict(l=10, r=20, t=10, b=10))
            st.plotly_chart(fig_h, use_container_width=True)
        else:
            st.info("Parâmetros insuficientes para heatmap.")

    with st.expander("Estrutura do Resultado Operacional Projetado (Barras Empilhadas)", expanded=False):
        st.info("Sem ΔRO por família consolidado nesta aba.")

# ============================
# TAB 4 — Insights Preditivos & Otimização (ML)
# ============================
with tab4:
    st.subheader("Previsão de Demanda (Volume) — Ano Corrente")

    fam_col = "Família Comercial"
    months_labels = [MONTHS_PT[m] for m in range(1, 13)]
    ytg_months = [m for m in range(cutoff + 1, 13)]
    ytd_months = [m for m in range(1, cutoff + 1)]

    # ---------- ML (YTG) — por família ----------
    ml_pred_map = _ml_predict_ytg_by_family_smart(df, year, ytg_months, fam_col=fam_col)

    # ---------- séries totais ----------
    real_ytd = df[(df["ano"] == year) & (df["cenario"] == re_scen) & df["indicador_id"].eq(VOLUME_ID) & df["mes"].isin(ytd_months)]
    tot_real_ytd_by_m = real_ytd.groupby("mes")["valor"].sum().reindex(range(1, cutoff + 1), fill_value=0.0)

    ml_total_ytg_by_m = {m: 0.0 for m in ytg_months}
    for (_fam, m), vhat in ml_pred_map.items():
        ml_total_ytg_by_m[m] = ml_total_ytg_by_m.get(m, 0.0) + float(vhat)

    # ---------- (NOVO) Projetado (YTG) — UI -> RES -> RE (totais mensais p/ Insights) ----------
    sim_month_totals = get_sim_month_totals(df, year)

    # Plot total (linha): Realizado YTD + Previsto (ML YTG) + Projetado (YTG)
    rows_plot = []
    for m in ytd_months:
        rows_plot.append({"Mês": MONTHS_PT[m], "Volume": float(tot_real_ytd_by_m.get(m, 0.0)), "Série": "Realizado (YTD)"})
    for m in ytg_months:
        rows_plot.append({"Mês": MONTHS_PT[m], "Volume": float(ml_total_ytg_by_m.get(m, 0.0)), "Série": "Previsto (ML YTG)"})
        rows_plot.append({"Mês": MONTHS_PT[m], "Volume": float(sim_month_totals.get(m, 0.0)), "Série": "Projetado (YTG)"})
    df_total_plot = pd.DataFrame(rows_plot)

    total_real_ytd = float(sum(tot_real_ytd_by_m.values))
    total_ml_ytg = float(sum(ml_total_ytg_by_m.values()))
    total_fy_ml = total_real_ytd + total_ml_ytg

    c1, c2, c3 = st.columns(3)
    c1.metric("YTD (Realizado) — Volume", f"{total_real_ytd:,.0f}".replace(",", "."))
    c2.metric("YTG (Previsto ML) — Volume", f"{total_ml_ytg:,.0f}".replace(",", "."))
    c3.metric(f"FY{str(year)[-2:]} (YTD + ML)", f"{total_fy_ml:,.0f}".replace(",", "."))

    fig_total = px.line(df_total_plot, x="Mês", y="Volume", color="Série", markers=True,
                        title="Volume Total (todas as famílias) — YTD Realizado + YTG Previsto (ML) + YTG Projetado")
    # deixa a série Projetado (YTG) em vermelho (linha destacada)
    fig_total.for_each_trace(lambda tr: tr.update(line=dict(color="red", width=3)) if tr.name == "Projetado (YTG)" else ())
    fig_total.update_layout(height=380, margin=dict(l=10, r=20, t=40, b=10))
    st.plotly_chart(fig_total, use_container_width=True)

    # ---------- Top-5: |ML − Planejado| ----------
    st.markdown("### Top-5 Famílias — Maior diferença entre **Previsto ML** e **Planejado (YTG)**")

    # Base planejada (o mesmo baseline enviado ao core)
    plan_ytg_by_fam_m = {}
    if not plan_long.empty:
        gplan = plan_long.groupby(["Família Comercial", "mes"])["volume"].sum()
        plan_ytg_by_fam_m = {(str(f), int(m)): float(v) for (f, m), v in gplan.items()}
        fonte_baseline = fonte_plan
    else:
        fonte_baseline = "RE (fallback)"
    st.caption(f"Baseline Planejado (YTG) nos gráficos de ML: **{fonte_baseline}** (idêntico ao usado no P&L).")

    if ml_pred_map and plan_ytg_by_fam_m:
        fams = sorted({f for (f, _m) in ml_pred_map.keys()} | {f for (f, _m) in plan_ytg_by_fam_m.keys()})
        deltas = []
        for fam in fams:
            ml_sum = sum(ml_pred_map.get((fam, m), 0.0) for m in ytg_months)
            plan_sum = sum(plan_ytg_by_fam_m.get((fam, m), 0.0) for m in ytg_months)
            deltas.append({"Familia": fam, "Δ (ML − Planejado)": float(ml_sum - plan_sum), "Δ abs": abs(ml_sum - plan_sum)})
        df_rank = pd.DataFrame(deltas).sort_values("Δ abs", ascending=False).head(5)
        rows_f = []
        for fam in df_rank["Familia"]:
            for m in ytg_months:
                rows_f.append({"Familia": fam, "Mês": MONTHS_PT[m], "Volume": float(ml_pred_map.get((fam, m), 0.0)), "Série": "Previsto (ML YTG)"})
                rows_f.append({"Familia": fam, "Mês": MONTHS_PT[m], "Volume": float(plan_ytg_by_fam_m.get((fam, m), 0.0)), "Série": "Planejado (YTG)"})
        df_fplot = pd.DataFrame(rows_f)
        fig_f = px.line(df_fplot, x="Mês", y="Volume", color="Série", facet_col="Familia", facet_col_wrap=2, markers=True,
                        title="Comparativo ML vs Planejado — YTG (Top-5 Famílias)")
        fig_f.update_layout(height=520, margin=dict(l=10, r=20, t=40, b=10))
        st.plotly_chart(fig_f, use_container_width=True)
    else:
        st.info("Necessário baseline Planejado (YTG) e histórico suficiente para o ML.")

    st.markdown("---")
    st.subheader("Otimização de Mix (Prescritivo) — ±% sobre o **Planejado (YTG)**")

    lim_pct = st.slider("Limite de ajuste por família (±%)", min_value=0, max_value=20, value=3, step=1,
                        help="Mantemos o volume total do **Planejado (YTG)** constante.")
    lim_frac = lim_pct / 100.0

    if plan_ytg_by_fam_m and ytg_months:
        familias_opt = sorted({f for (f, _m) in plan_ytg_by_fam_m.keys()})
        v_cur_series = pd.Series({f: float(sum(plan_ytg_by_fam_m.get((f, m), 0.0) for m in ytg_months)) for f in familias_opt}, name="Planejado (YTG) Atual")
        v_cur = {f: float(v_cur_series.get(f, 0.0)) for f in familias_opt}
        Vtot = sum(v_cur.values())

        params = load_params()
        rbuc = {f: np.mean([_rb_uc(params, f, m) for m in ytg_months]) if ytg_months else 0.0 for f in familias_opt}
        rluc = {f: rl_uc_from_rb(rbuc[f]) for f in familias_opt}
        varuc = {f: np.mean([_ins_uc(params, f, m) + _toll_uc(params, f, m) + _frete_uc(params, f, m)]) if ytg_months else 0.0 for f in familias_opt}
        slope_ro_uc = {f: (rluc[f] - varuc[f]) - dme_pct * rluc[f] for f in familias_opt}

        model = pulp.LpProblem("MixMaxRO_PlanejadoYTG", pulp.LpMaximize)
        x = {f: pulp.LpVariable(f"x_{i}", lowBound=0) for i, f in enumerate(familias_opt)}
        model += pulp.lpSum([slope_ro_uc[f] * x[f] for f in familias_opt])
        model += pulp.lpSum([x[f] for f in familias_opt]) == Vtot
        for f in familias_opt:
            lo = max(0.0, (1.0 - lim_frac) * v_cur[f])
            hi = (1.0 + lim_frac) * v_cur[f]
            model += x[f] >= lo
            model += x[f] <= hi

        res = model.solve(pulp.PULP_CBC_CMD(msg=False))
        if res != 1:
            st.warning("Otimização não encontrou solução ótima.")
        else:
            mix_opt = {f: x[f].value() for f in familias_opt}
            df_mix = pd.DataFrame({"Familia": familias_opt,
                                   "Planejado (YTG) Atual": [v_cur[f] for f in familias_opt],
                                   "Mix Otimizado": [mix_opt[f] for f in familias_opt]})
            ro_atual = sum(slope_ro_uc[f] * v_cur[f] for f in familias_opt)
            ro_otimo = sum(slope_ro_uc[f] * mix_opt[f] for f in familias_opt)
            ganho = ro_otimo - ro_atual
            st.metric("Ganho Potencial de RO (YTG)", f"{ganho:,.0f}".replace(",", "."), help="Mantendo o volume total Planejado (YTG).")
            fig_mix = go.Figure()
            fig_mix.add_bar(name="Planejado (YTG) Atual", x=df_mix["Familia"], y=df_mix["Planejado (YTG) Atual"])
            fig_mix.add_bar(name="Mix Otimizado", x=df_mix["Familia"], y=df_mix["Mix Otimizado"])
            fig_mix.update_layout(barmode="group", height=420, margin=dict(l=10, r=20, t=10, b=10))
            st.plotly_chart(fig_mix, use_container_width=True)
    else:
        st.info("Para otimizar o mix é necessário ter baseline Planejado (YTG).")

    st.markdown("---")
    st.subheader("Série 2020 → 2030 — Volume, Receita Líquida e Resultado Operacional (Histórico + ML)")

    def _annual_forecast_2020_2030(df: pd.DataFrame, indicator_id: str, current_year: int, end_year: int = 2030, start_year: int = 2020) -> pd.DataFrame:
        hist_mask = (df["indicador_id"].eq(indicator_id) & df["cenario"].apply(lambda s: _is_realizado(str(s))))
        hist = df[hist_mask].groupby(["ano", "mes"])["valor"].sum().reset_index()
        if hist.empty:
            years = list(range(start_year, end_year + 1))
            return pd.DataFrame({"Ano": years, "Valor": [0.0]*len(years)})
        last_hist = int(hist["ano"].max())
        years_all = list(range(start_year, end_year + 1))
        anual_hist = hist.groupby("ano")["valor"].sum().rename("Total").reset_index()
        months = list(range(1, 13))
        hist = hist.sort_values(["ano", "mes"]).reset_index(drop=True)
        def lag12(a, m):
            v = hist[(hist["ano"] == a - 1) & (hist["mes"] == m)]["valor"]
            return float(v.values[0]) if len(v) else np.nan
        hist["lag12"] = [lag12(int(r.ano), int(r.mes)) for r in hist.itertuples()]
        hist["lag12"] = hist.groupby("mes")["lag12"].transform(lambda s: s.fillna(s.mean()))
        hist["lag12"] = hist["lag12"].fillna(hist["valor"].mean())
        X = pd.DataFrame({
            "mes": hist["mes"].astype(int),
            "sin": np.sin(2 * np.pi * hist["mes"].astype(float) / 12.0),
            "cos": np.cos(2 * np.pi * hist["mes"].astype(float) / 12.0),
            "ano": hist["ano"].astype(int),
            "lag12": hist["lag12"].astype(float),
        })
        y = hist["valor"].astype(float).values
        if len(anual_hist) >= 2:
            a0, a1 = anual_hist.iloc[0]["Total"], anual_hist.iloc[-1]["Total"]
            n = int(anual_hist.iloc[-1]["ano"] - anual_hist.iloc[0]["ano"])
            cagr = (a1 / a0) ** (1 / n) - 1 if (a0 > 0 and n > 0) else 0.0
        else:
            cagr = 0.0
        def predict_year(y_target: int) -> float:
            Xf = pd.DataFrame({
                "mes": months,
                "sin": np.sin(2 * np.pi * np.array(months) / 12.0),
                "cos": np.cos(2 * np.pi * np.array(months) / 12.0),
                "ano": [y_target] * 12,
                "lag12": [lag12(y_target, m) if not np.isnan(lag12(y_target, m)) else hist[hist["mes"]==m]["valor"].mean() for m in months],
            })
            yh = _safe_gbr_train_pred(X, y, Xf)
            total = float(np.maximum(yh, 0.0).sum())
            if total <= 0.0 and len(anual_hist) >= 1:
                base = float(anual_hist[anual_hist["ano"] == last_hist]["Total"].values[0]) if (last_hist in anual_hist["ano"].values) else float(anual_hist["Total"].iloc[-1])
                years_ahead = y_target - last_hist
                total = base * ((1 + cagr) ** years_ahead)
            return total
        rows = []
        for y_out in years_all:
            if y_out <= last_hist:
                val = float(anual_hist[anual_hist["ano"] == y_out]["Total"].sum())
            else:
                val = predict_year(y_out)
            rows.append({"Ano": y_out, "Valor": val})
        return pd.DataFrame(rows)

    df_vol_2030 = _annual_forecast_2020_2030(df, VOLUME_ID, current_year=year, end_year=2030, start_year=2020)
    df_rl_2030  = _annual_forecast_2020_2030(df, "receita_liquida", current_year=year, end_year=2030, start_year=2020)
    df_ro_2030  = _annual_forecast_2020_2030(df, "resultado_operacional", current_year=year, end_year=2030, start_year=2020)

    plot_2030 = pd.DataFrame({
        "Ano": df_vol_2030["Ano"],
        "Volume (UC)": df_vol_2030["Valor"],
        "Receita Líquida": df_rl_2030["Valor"],
        "Resultado Operacional": df_ro_2030["Valor"],
    })
    long_2030 = plot_2030.melt(id_vars="Ano", var_name="Indicador", value_name="Valor")
    fig2030 = px.line(long_2030, x="Ano", y="Valor", color="Indicador", markers=True, title="Histórico (Realizado) + Previsão (ML) até 2030")
    fig2030.update_layout(height=420, margin=dict(l=10, r=20, t=40, b=10), xaxis=dict(dtick=1))
    st.plotly_chart(fig2030, use_container_width=True)

# --------------------------------------------------------------------------------------
# Rodapé — Premissas do Modelo
# --------------------------------------------------------------------------------------
with st.expander("Premissas do Modelo", expanded=False):
    st.markdown(
        """
- **Custos Variáveis** (Insumo, Frete) escalam linearmente com o volume.  
- **Impostos sobre Receita** (Convênio) são calculados como uma **% fixa da Receita Bruta** (25,6%).  
- **Despesas com Marketing (DME)** são calculadas como uma **% fixa da Receita Líquida**.  
- **⚠️ Custos de EI e Opex/Chargeback/Depreciação** são considerados **Não Alocados** e **não escalam** com volume nesta simulação; refletem valores do BP/Realizado.  
        """
    )
