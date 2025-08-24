# pages/04_POH.py
# --------------------------------------------------------------------------------------
# POH — Abas:
#   (1) Resumo por Tecnologia (YTG)
#   (2) Tecnologia × Mês (YTG)
#   (3) Famílias (Resumo e Composição)
#   (4) Rolagem de Estoque (YTG)  <- NOVA
#
# Regras:
#  - Estoque inicial considerado 0 (temporário)
#  - Sem gráficos por ora
#  - Números aplicam a escala (1x / 1.000x / 1.000.000x), sem exibir o texto da escala
# Paths (somente relativos):
#  - data/parquet/current.parquet
#  - data/parquet/res_working.parquet
#  - data/cap_poh/capacidade_tecnologias.* (xlsx/csv/parquet)
#  - data/estoque/inicial/  (placeholder, não utilizado ainda)
#  - data/estoque/politica/ (placeholder, não utilizado ainda)
# --------------------------------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# ===============================
# Config & Consts
# ===============================
st.set_page_config(page_title="POH", page_icon="🏭", layout="wide")

DATA_DIR = Path("data")
PARQUET_DIR = DATA_DIR / "parquet"
CAP_DIR = DATA_DIR / "cap_poh"
ESTOQUE_INICIAL_DIR = DATA_DIR / "estoque" / "inicial"   # placeholder
POLITICA_ESTOQUE_DIR = DATA_DIR / "estoque" / "politica" # placeholder

MONTHS_PT = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}

# ===============================
# Utilitários gerais (robustos)
# ===============================
def _dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas duplicadas mantendo a primeira ocorrência."""
    if not isinstance(df, pd.DataFrame):
        return df
    return df.loc[:, ~df.columns.duplicated()].copy()

def _to_num(x):
    """Coerção numérica robusta (Series/DataFrame/escalar) -> Series/float com NaN->0."""
    try:
        import pandas as pd
        import numpy as np
        if isinstance(x, (int, float, np.number)):
            return float(x)
        if isinstance(x, pd.DataFrame):
            if x.shape[1] == 0:
                return pd.Series([], dtype=float)
            x = x.iloc[:, 0]
        return pd.to_numeric(x, errors="coerce").fillna(0.0)
    except Exception:
        try:
            return pd.to_numeric(x, errors="coerce").fillna(0.0)
        except Exception:
            return 0.0

def _scale_factor(lbl: str) -> int:
    return {"1x":1, "1.000x":1_000, "1.000.000x":1_000_000}.get(lbl, 1)

def _safe_selectbox(label, options, index=0, key=None):
    try:
        return st.selectbox(label, options=options, index=index, key=key)
    except Exception:
        return st.selectbox(label, options=options, key=key)

def _read_parquet_safe(path: Path) -> pd.DataFrame:
    try:
        if not path.exists():
            st.warning(f"Arquivo não encontrado: `{path.as_posix()}`.")
            return pd.DataFrame()
        return pd.read_parquet(path)
    except Exception as e:
        st.error(f"Falha ao ler parquet `{path.as_posix()}`: {e}")
        return pd.DataFrame()

# ===============================
# Loaders (fallback) — compatíveis com schema descrito
# ===============================
@st.cache_data(ttl=600, show_spinner=False)
def _load_current_fb() -> pd.DataFrame:
    df = _read_parquet_safe(PARQUET_DIR / "current.parquet")
    if df.empty:
        return df
    ren = {}
    for c in df.columns:
        cl = str(c).lower().strip()
        if cl in {"familia comercial","família comercial","familia","familia_comercial"}: ren[c] = "Família Comercial"
        if cl in {"tecnologia","tec"}: ren[c] = "Tecnologia"
    if ren:
        df = df.rename(columns=ren)
    if "ano" in df.columns: df["ano"] = pd.to_numeric(df["ano"], errors="coerce").fillna(0).astype(int)
    if "mes" in df.columns: df["mes"] = pd.to_numeric(df["mes"], errors="coerce").fillna(0).astype(int)
    if "volume_uc" in df.columns: df["volume_uc"] = pd.to_numeric(df["volume_uc"], errors="coerce").fillna(0.0)
    if "valor" in df.columns: df["valor"] = pd.to_numeric(df["valor"], errors="coerce").fillna(0.0)
    return _dedup_columns(df)

@st.cache_data(ttl=600, show_spinner=False)
def _load_res_fb() -> pd.DataFrame:
    df = _read_parquet_safe(PARQUET_DIR / "res_working.parquet")
    if df.empty:
        return df
    if "volume_uc" not in df.columns:
        for c in df.columns:
            if str(c).lower().strip() in {"volume","vol_uc","volumes","vol"}:
                df = df.rename(columns={c: "volume_uc"})
                break
    ren = {}
    for c in df.columns:
        cl = str(c).lower().strip()
        if cl in {"familia comercial","família comercial","familia","familia_comercial"}: ren[c] = "Família Comercial"
        if cl in {"tecnologia","tec"}: ren[c] = "Tecnologia"
    if ren:
        df = df.rename(columns=ren)
    if "ano" in df.columns: df["ano"] = pd.to_numeric(df["ano"], errors="coerce").fillna(0).astype(int)
    if "mes" in df.columns: df["mes"] = pd.to_numeric(df["mes"], errors="coerce").fillna(0).astype(int)
    df["volume_uc"] = pd.to_numeric(df.get("volume_uc", 0.0), errors="coerce").fillna(0.0)
    return _dedup_columns(df)

@st.cache_data(ttl=600, show_spinner=False)
def _load_cap_fb() -> pd.DataFrame:
    try:
        cands = sorted(CAP_DIR.glob("capacidade_tecnologias*.*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            st.warning("Nenhum arquivo de capacidade encontrado em `data/cap_poh/`.")
            return pd.DataFrame(columns=["Tecnologia","ano","mes","capacidade"])
        path = cands[0]
        if path.suffix.lower() in {".xlsx",".xls"}:
            raw = pd.read_excel(path)
        elif path.suffix.lower() == ".csv":
            raw = pd.read_csv(path)
        else:
            raw = pd.read_parquet(path)
    except Exception as e:
        st.error(f"Falha ao carregar capacidade: {e}")
        return pd.DataFrame(columns=["Tecnologia","ano","mes","capacidade"])

    raw.columns = [str(c).strip().lower() for c in raw.columns]
    try:
        if {"tecnologia","ano","mes","capacidade"}.issubset(raw.columns):
            df = raw[["tecnologia","ano","mes","capacidade"]].copy()
            df["ano"] = pd.to_numeric(df["ano"], errors="coerce").fillna(0).astype(int)
            df["mes"] = pd.to_numeric(df["mes"], errors="coerce").fillna(0).astype(int)
            df["capacidade"] = pd.to_numeric(df["capacidade"], errors="coerce").fillna(0.0)
            return _dedup_columns(df.rename(columns={"tecnologia":"Tecnologia"}))

        meses_map = {"jan":1,"fev":2,"mar":3,"abr":4,"mai":5,"jun":6,"jul":7,"ago":8,"set":9,"out":10,"nov":11,"dez":12}
        if {"tecnologia","ano"}.issubset(raw.columns):
            mes_cols = [c for c in raw.columns if c[:3] in meses_map]
            if mes_cols:
                df = raw[["tecnologia","ano"] + mes_cols].copy()
                df = df.melt(id_vars=["tecnologia","ano"], var_name="mes_nome", value_name="capacidade")
                df["mes"] = df["mes_nome"].str[:3].map(meses_map).astype(int)
                df["ano"] = pd.to_numeric(df["ano"], errors="coerce").fillna(0).astype(int)
                df["capacidade"] = pd.to_numeric(df["capacidade"], errors="coerce").fillna(0.0)
                df = df[["tecnologia","ano","mes","capacidade"]].rename(columns={"tecnologia":"Tecnologia"})
                return _dedup_columns(df)

        if {"tecnologia","capacidade"}.issubset(raw.columns):
            from datetime import datetime
            ano_atual = datetime.now().year
            base = raw[["tecnologia","capacidade"]].copy()
            base["capacidade"] = pd.to_numeric(base["capacidade"], errors="coerce").fillna(0.0)
            base["key"] = 1
            meses = pd.DataFrame({"mes": list(range(1,13)), "key":1})
            df = base.merge(meses, on="key").drop(columns="key").assign(ano=ano_atual)
            df = df.rename(columns={"tecnologia":"Tecnologia"})[["Tecnologia","ano","mes","capacidade"]]
            return _dedup_columns(df)

        st.warning("Arquivo de capacidade não reconhecido. Esperado: (tecnologia, ano, mes, capacidade) "
                   "ou wide Jan..Dez / constante por tecnologia.")
        return pd.DataFrame(columns=["Tecnologia","ano","mes","capacidade"])
    except Exception as e:
        st.error(f"Falha ao padronizar capacidade: {e}")
        return pd.DataFrame(columns=["Tecnologia","ano","mes","capacidade"])

# ===============================
# Funções de negócio (fallback)
# ===============================
def _compute_cutoff_for_year_fb(df_current: pd.DataFrame, ano: int) -> int:
    try:
        if df_current.empty or "ano" not in df_current.columns or "mes" not in df_current.columns:
            return 0
        d = df_current[df_current["ano"] == ano].copy()
        if d.empty:
            return 0
        vol = pd.to_numeric(d.get("volume_uc", d.get("valor", 0.0)), errors="coerce").fillna(0.0)
        m = d.assign(_v=vol).query("_v>0")["mes"]
        return int(m.max()) if not m.empty else 0
    except Exception:
        return 0

def _familia_tecnologia_map_fb(df_current: pd.DataFrame, ano: int) -> pd.DataFrame:
    try:
        if df_current.empty:
            return pd.DataFrame(columns=["Família Comercial","Tecnologia"])
        need = {"Família Comercial","Tecnologia","ano"}
        if not need.issubset(set(df_current.columns)):
            return pd.DataFrame(columns=["Família Comercial","Tecnologia"])
        out = (df_current[df_current["ano"] == ano][["Família Comercial","Tecnologia"]]
               .dropna().drop_duplicates())
        return _dedup_columns(out)
    except Exception:
        return pd.DataFrame(columns=["Família Comercial","Tecnologia"])

def _build_demanda_ytg_fb(df_res: pd.DataFrame, df_map: pd.DataFrame, ano: int, cutoff: int) -> pd.DataFrame:
    try:
        if df_res.empty:
            return pd.DataFrame(columns=["Tecnologia","ano","mes","demanda"])
        r = df_res[(df_res["ano"] == ano) & (df_res["mes"] > cutoff)].copy()
        if r.empty:
            return pd.DataFrame(columns=["Tecnologia","ano","mes","demanda"])
        base = r
        if "Tecnologia" not in base.columns and "Família Comercial" in base.columns and not df_map.empty:
            base = base.merge(df_map, on="Família Comercial", how="left")
        if "Tecnologia" not in base.columns:
            base["Tecnologia"] = "(N/A)"
        base["volume_uc"] = pd.to_numeric(base.get("volume_uc", 0.0), errors="coerce").fillna(0.0)
        out = (base.groupby(["Tecnologia","ano","mes"], as_index=False)["volume_uc"].sum()
               .rename(columns={"volume_uc":"demanda"}))
        return _dedup_columns(out)
    except Exception as e:
        st.error(f"Falha ao computar demanda YTG: {e}")
        return pd.DataFrame(columns=["Tecnologia","ano","mes","demanda"])

# ===============================
# Import do core/calculator_poh (se disponível)
# ===============================
try:
    from core.calculator_prod import (
        load_current,
        load_res_working,
        load_capacidade_tecnologias,
        compute_cutoff_for_year,
        familia_tecnologia_map,
        build_demanda_ytg,
        build_cap_vs_demanda,
        MONTHS_PT as MONTHS_PT_CORE,
    )
    HAVE_CORE = True
    MONTHS_PT = MONTHS_PT_CORE or MONTHS_PT
except Exception:
    HAVE_CORE = False

# ===============================
# Carregamento dos dados
# ===============================
try:
    cur = load_current() if HAVE_CORE else _load_current_fb()
    res = load_res_working() if HAVE_CORE else _load_res_fb()
    cap = load_capacidade_tecnologias() if HAVE_CORE else _load_cap_fb()
except Exception as e:
    st.error(f"Falha ao carregar dados: {e}")
    cur, res, cap = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Ano & Escala UI (robusto)
try:
    anos = sorted(set(cur.get("ano", pd.Series(dtype=int)).dropna().tolist()) |
                  set(res.get("ano", pd.Series(dtype=int)).dropna().tolist()))
    ano_default = anos[-1] if anos else pd.Timestamp.today().year
    c1, c2 = st.columns([1,1])
    with c1:
        ano = _safe_selectbox("Ano", options=anos or [ano_default], index=(len(anos)-1 if anos else 0))
    with c2:
        st.session_state.setdefault("poh_scale_label", "1x")
        escala_label = _safe_selectbox("Escala", ["1x","1.000x","1.000.000x"],
                                       index=["1x","1.000x","1.000.000x"].index(st.session_state["poh_scale_label"]),
                                       key="poh_scale_label")
    scale = _scale_factor(escala_label)
except Exception as e:
    st.error(f"Falha na seleção de Ano/Escala: {e}")
    ano = pd.Timestamp.today().year
    scale = 1

# Cutoff
try:
    cutoff = compute_cutoff_for_year(cur, ano) if HAVE_CORE else _compute_cutoff_for_year_fb(cur, ano)
except Exception:
    cutoff = 0

st.caption(f"Cutoff YTD: **mês {cutoff or 'n/d'}** · YTG = meses > {cutoff}. Estoque inicial = **0** (temporário).")

# Mapa Família→Tecnologia e Demanda YTG
try:
    fam_tec = familia_tecnologia_map(cur, ano) if HAVE_CORE else _familia_tecnologia_map_fb(cur, ano)
except Exception as e:
    st.error(f"Falha ao mapear Família→Tecnologia: {e}")
    fam_tec = pd.DataFrame(columns=["Família Comercial","Tecnologia"])

try:
    demanda_ytg = build_demanda_ytg(res, fam_tec, ano, cutoff) if HAVE_CORE else _build_demanda_ytg_fb(res, fam_tec, ano, cutoff)
except Exception as e:
    st.error(f"Falha ao construir Demanda YTG: {e}")
    demanda_ytg = pd.DataFrame(columns=["Tecnologia","ano","mes","demanda"])

# Capacidade vs Demanda (Tec×Mês)
try:
    if HAVE_CORE:
        cap_vs_dem, _ = build_cap_vs_demanda(ano, cur, res, cap)
    else:
        cap_y = cap.copy()
        if not cap_y.empty:
            if "ano" in cap_y.columns:
                cap_y = cap_y[cap_y["ano"] == ano].copy()
            else:
                cap_y["ano"] = ano
        cap_vs_dem = demanda_ytg.merge(
            cap_y[["Tecnologia","ano","mes","capacidade"]] if not cap_y.empty else
            pd.DataFrame(columns=["Tecnologia","ano","mes","capacidade"]),
            on=["Tecnologia","ano","mes"], how="left"
        )
        cap_vs_dem = _dedup_columns(cap_vs_dem)
        cap_vs_dem["capacidade"] = _to_num(cap_vs_dem.get("capacidade", 0.0))
        cap_vs_dem["demanda"] = _to_num(cap_vs_dem.get("demanda", 0.0))
        cap_vs_dem["cobertura_pct"] = np.where(cap_vs_dem["demanda"]>0,
                                               cap_vs_dem["capacidade"]/cap_vs_dem["demanda"],
                                               np.nan)
        cap_vs_dem["critico"] = cap_vs_dem["capacidade"] < cap_vs_dem["demanda"]
        cap_vs_dem["mes_nome"] = cap_vs_dem["mes"].map(MONTHS_PT)
except Exception as e:
    st.error(f"Falha ao compor Capacidade vs Demanda: {e}")
    cap_vs_dem = pd.DataFrame(columns=["Tecnologia","ano","mes","demanda","capacidade","cobertura_pct","critico","mes_nome"])

st.divider()

# ===============================
# Abas
# ===============================
tab1, tab2, tab3, tab4 = st.tabs([
    "Resumo por Tecnologia (YTG)",
    "Tecnologia × Mês (YTG)",
    "Famílias (Resumo e Composição)",
    "Rolagem de Estoque (YTG)"  # NOVA
])

# -------------------------------
# Aba 1: Resumo por Tecnologia
# -------------------------------
with tab1:
    st.subheader("Resumo por Tecnologia (YTG)")
    try:
        if cap_vs_dem.empty:
            st.info("Sem dados de Capacidade vs Demanda para o ano/meses selecionados.")
        else:
            dem_sum = (cap_vs_dem.groupby("Tecnologia", as_index=False)["demanda"]
                                  .sum().rename(columns={"demanda":"demanda_ytg"}))
            cap_sum = (cap_vs_dem.groupby("Tecnologia", as_index=False)["capacidade"]
                                  .sum().rename(columns={"capacidade":"capacidade_ytg"}))
            df_sum = _dedup_columns(dem_sum.merge(cap_sum, on="Tecnologia", how="outer").fillna(0.0))

            dem_s = _to_num(df_sum["demanda_ytg"])
            cap_s = _to_num(df_sum["capacidade_ytg"])

            df_sum["Demanda YTG"]    = (dem_s / scale).round(0)
            df_sum["Capacidade YTG"] = (cap_s / scale).round(0)
            df_sum["Gap (Cap−Dem)"]  = ((cap_s - dem_s) / scale).round(0)

            cov_num = np.where(dem_s > 0, cap_s / dem_s, np.nan)
            df_sum["Cobertura %"] = pd.Series(cov_num, index=df_sum.index).map(
                lambda x: "-" if pd.isna(x) else f"{x*100:,.1f}%".replace(",", ".")
            )

            out = df_sum[["Tecnologia","Demanda YTG","Capacidade YTG","Gap (Cap−Dem)","Cobertura %"]]
            st.dataframe(out.sort_values(["Tecnologia"]), use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"[Resumo por Tecnologia] Erro: {e}")

# -------------------------------
# Aba 2: Tecnologia × Mês
# -------------------------------
with tab2:
    st.subheader("Tecnologia × Mês — Capacidade vs Demanda (YTG)")
    try:
        if cap_vs_dem.empty:
            st.info("Sem dados para o ano/meses selecionados.")
        else:
            df2 = cap_vs_dem.copy()
            df2 = _dedup_columns(df2)
            df2["demanda"] = _to_num(df2.get("demanda", 0.0))
            df2["capacidade"] = _to_num(df2.get("capacidade", 0.0))

            df2["Demanda"]    = (df2["demanda"] / scale).round(0)
            df2["Capacidade"] = (df2["capacidade"] / scale).round(0)
            df2["Gap (Cap−Dem)"] = ((df2["capacidade"] - df2["demanda"]) / scale).round(0)

            df2["Cobertura %"] = df2.get("cobertura_pct", np.nan).map(
                lambda x: "-" if pd.isna(x) else f"{float(x)*100:,.1f}%".replace(",", ".")
            )
            df2["Crítico"] = np.where(df2.get("critico", False), "Sim", "Não")
            df2 = (df2.rename(columns={"mes":"Mês Num","mes_nome":"Mês"})
                     [["Tecnologia","ano","Mês Num","Mês","Demanda","Capacidade","Gap (Cap−Dem)","Cobertura %","Crítico"]]
                     .sort_values(["Tecnologia","ano","Mês Num"], kind="mergesort"))
            st.dataframe(df2, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"[Tecnologia × Mês] Erro: {e}")

# -------------------------------
# Aba 3: Famílias (Resumo e Composição)
# -------------------------------
with tab3:
    st.subheader("Famílias — Resumo e Composição (YTG)")
    try:
        res_y = res[(res.get("ano") == ano) & (res.get("mes") > cutoff)].copy() if not res.empty else pd.DataFrame()
        if res_y.empty and not res.empty:
            res_y = res[res.get("ano") == ano].copy()

        if res_y.empty or "Família Comercial" not in res_y.columns:
            st.info("Sem dados suficientes (res_working ano/mês ou coluna 'Família Comercial').")
        else:
            fam_map = _dedup_columns(fam_tec.copy())

            # Resumo por Família (demanda do UI)
            fam_resumo = (res_y.groupby(["Família Comercial"], as_index=False)["volume_uc"]
                             .sum().rename(columns={"volume_uc":"Demanda YTG (UC)"}))

            # Capacidade por Família = soma das capacidades das tecnologias associadas (YTG)
            cap_fam = fam_map.merge(
                cap_vs_dem[["Tecnologia","ano","mes","capacidade"]] if not cap_vs_dem.empty else
                pd.DataFrame(columns=["Tecnologia","ano","mes","capacidade"]),
                on="Tecnologia", how="left"
            )
            cap_fam = _dedup_columns(cap_fam)
            if not cap_fam.empty:
                cap_fam = cap_fam.query("ano == @ano and mes > @cutoff")

            if not cap_fam.empty:
                cap_fam_sum = (cap_fam.groupby("Família Comercial", as_index=False)["capacidade"]
                                      .sum().rename(columns={"capacidade":"Capacidade YTG (UC)"}))
            else:
                cap_fam_sum = pd.DataFrame({"Família Comercial": fam_resumo["Família Comercial"], "Capacidade YTG (UC)": 0.0})

            fam_join = _dedup_columns(fam_resumo.merge(cap_fam_sum, on="Família Comercial", how="left"))

            dem_f = _to_num(fam_join["Demanda YTG (UC)"])
            cap_f = _to_num(fam_join["Capacidade YTG (UC)"])
            cov_num = np.where(dem_f > 0, cap_f / dem_f, np.nan)

            fam_join["Demanda YTG (UC)"]   = (dem_f / scale).round(0)
            fam_join["Capacidade YTG (UC)"] = (cap_f / scale).round(0)
            fam_join["_cov_sort"] = pd.Series(cov_num, index=fam_join.index).fillna(np.inf)
            fam_join["Cobertura %"] = pd.Series(cov_num, index=fam_join.index).map(
                lambda x: "-" if pd.isna(x) else f"{x*100:,.1f}%".replace(",", ".")
            )

            fam_sorted = fam_join.sort_values(by=["_cov_sort","Família Comercial"], kind="mergesort")
            st.markdown("**Resumo por Família**")
            st.dataframe(
                fam_sorted.drop(columns=["_cov_sort"]),
                use_container_width=True, hide_index=True
            )

            # Composição — Família × Tecnologia × Mês
            st.markdown("**Composição — Família × Tecnologia × Mês**")
            comp = (res_y.merge(fam_map, on="Família Comercial", how="left")
                        .merge(cap_vs_dem[["Tecnologia","ano","mes","capacidade","demanda"]] if not cap_vs_dem.empty else
                               pd.DataFrame(columns=["Tecnologia","ano","mes","capacidade","demanda"]),
                               on=["Tecnologia","ano","mes"], how="left"))
            comp = _dedup_columns(comp)
            comp["volume_uc"] = _to_num(comp.get("volume_uc", 0.0))
            comp["capacidade"] = _to_num(comp.get("capacidade", 0.0))
            comp["demanda"] = _to_num(comp.get("demanda", 0.0))
            comp["Demanda (UC)"] = (comp["volume_uc"] / scale).round(0)
            comp["Capacidade (UC)"] = (comp["capacidade"] / scale).round(0)
            comp["Mês"] = comp.get("mes", pd.Series(dtype=int)).map(MONTHS_PT)
            comp["Crítico"] = np.where((comp["capacidade"] < comp["demanda"]), "Sim", "Não")
            cols = ["Família Comercial","Tecnologia","ano","mes","Mês","Demanda (UC)","Capacidade (UC)","Crítico"]
            cols = [c for c in cols if c in comp.columns]
            comp = comp[cols].sort_values(["Família Comercial","Tecnologia","ano","mes"], kind="mergesort")
            st.dataframe(comp, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"[Famílias] Erro: {e}")

# -------------------------------
# Aba 4: Rolagem de Estoque (YTG) — PROJEÇÃO ROBUSTA + ROLAGEM SEQUENCIAL + LAYOUT WIDE
# -------------------------------
with tab4:
    st.subheader("Rolagem de Estoque (YTG)")
    try:
        # Helpers: recorte YTG com fallback para FY; grade Família×Mês resiliente
        def _slice_ytg_or_fy(df: pd.DataFrame, ano_sel: int, cutoff_m: int) -> pd.DataFrame:
            if df is None or df.empty or not {"ano","mes"}.issubset(df.columns):
                return pd.DataFrame()
            d = df[df["ano"] == ano_sel].copy()
            if d.empty:
                return pd.DataFrame()
            ytg = d[d["mes"] > cutoff_m].copy()
            return ytg if not ytg.empty else d

        def _build_fam_mes_grid(ui_df: pd.DataFrame, re_df: pd.DataFrame, fam_map_df: pd.DataFrame, cutoff_m: int) -> pd.DataFrame:
            cols = ["Família Comercial","mes"]
            parts = []
            if isinstance(ui_df, pd.DataFrame) and not ui_df.empty and set(cols).issubset(ui_df.columns):
                parts.append(ui_df[cols])
            if isinstance(re_df, pd.DataFrame) and not re_df.empty and set(cols).issubset(re_df.columns):
                parts.append(re_df[cols])
            grid = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
            grid = grid.dropna().drop_duplicates()
            if grid.empty:
                fams = set()
                if isinstance(ui_df, pd.DataFrame) and "Família Comercial" in ui_df.columns:
                    fams |= set(ui_df["Família Comercial"].dropna().unique().tolist())
                if isinstance(re_df, pd.DataFrame) and "Família Comercial" in re_df.columns:
                    fams |= set(re_df["Família Comercial"].dropna().unique().tolist())
                if not fams and isinstance(fam_map_df, pd.DataFrame) and "Família Comercial" in fam_map_df.columns:
                    fams |= set(fam_map_df["Família Comercial"].dropna().unique().tolist())
                fams = sorted(fams)
                meses_ytg = list(range(min(cutoff_m + 1, 12), 13)) if cutoff_m < 12 else []
                meses = meses_ytg if meses_ytg else list(range(1, 13))
                grid = pd.DataFrame([(f, m) for f in fams for m in meses], columns=cols)
            return grid

        # Toggle de projeção (RES working) — espelha a DRE
        default_proj = bool(st.session_state.get("dre_use_projection", True))
        usar_proj = st.checkbox("Usar Projeção de Vendas (RES working) para YTG",
                                value=default_proj, key="poh_use_projection")

        # Slices com fallback YTG→FY
        res_base = _slice_ytg_or_fy(res, ano, cutoff)
        cur_base = _slice_ytg_or_fy(cur, ano, cutoff)

        # UI (produção provisória = UI do RES)
        ui_agg = pd.DataFrame(columns=["Família Comercial","mes","UI (UC)"])
        if not res_base.empty and {"Família Comercial","mes","volume_uc"}.issubset(res_base.columns):
            ui_agg = (res_base.groupby(["Família Comercial","mes"], as_index=False)["volume_uc"]
                              .sum().rename(columns={"volume_uc":"UI (UC)"}))

        # RE (vendas quando não usar projeção)
        re_agg = pd.DataFrame(columns=["Família Comercial","mes","RE (UC)"])
        if not cur_base.empty and {"Família Comercial","mes","volume_uc"}.issubset(cur_base.columns):
            re_agg = (cur_base.groupby(["Família Comercial","mes"], as_index=False)["volume_uc"]
                              .sum().rename(columns={"volume_uc":"RE (UC)"}))

        # Fallback de projeção: se usar_proj=True mas ui_agg está vazio, cair para RE
        if usar_proj and ui_agg.empty and not re_agg.empty:
            st.info("Projeção selecionada, porém RES working está vazio para o recorte. Usando RE como fallback temporário.")
            ui_agg = re_agg.rename(columns={"RE (UC)":"UI (UC)"})

        # Grade Família×Mês
        fam_mes = _build_fam_mes_grid(ui_agg, re_agg, fam_tec, cutoff)

        if fam_mes.empty:
            st.info("Sem famílias/meses para rolagem (dados de UI/RE e mapeamento ausentes).")
        else:
            # Produção = UI (por enquanto)
            prod = fam_mes.merge(ui_agg, on=["Família Comercial","mes"], how="left")
            prod["UI (UC)"] = _to_num(prod.get("UI (UC)", 0.0))
            prod = prod.rename(columns={"UI (UC)":"Produção (UC)"})

            # Vendas = UI (se projeção) OU RE
            if usar_proj:
                vendas = fam_mes.merge(ui_agg, on=["Família Comercial","mes"], how="left") \
                                .rename(columns={"UI (UC)":"Vendas (UC)"})
            else:
                vendas = fam_mes.merge(re_agg, on=["Família Comercial","mes"], how="left") \
                                .rename(columns={"RE (UC)":"Vendas (UC)"})
            vendas["Vendas (UC)"] = _to_num(vendas.get("Vendas (UC)", 0.0))

            # Join Produção × Vendas
            est = fam_mes.merge(prod[["Família Comercial","mes","Produção (UC)"]], on=["Família Comercial","mes"], how="left")
            est = est.merge(vendas[["Família Comercial","mes","Vendas (UC)"]], on=["Família Comercial","mes"], how="left")
            est = _dedup_columns(est)
            est["Produção (UC)"] = _to_num(est.get("Produção (UC)", 0.0))
            est["Vendas (UC)"]    = _to_num(est.get("Vendas (UC)", 0.0))

            # ===== Rolagem sequencial =====
            est = est.sort_values(["Família Comercial","mes"], kind="mergesort")
            est["Δ (UC)"] = est["Produção (UC)"] - est["Vendas (UC)"]
            # Estoque Final acumulado (inicia em 0 por família)
            est["Estoque Final (UC)"] = est.groupby("Família Comercial")["Δ (UC)"].cumsum()
            # Estoque Inicial é o Estoque Final do mês anterior (shift), com 0 no primeiro mês
            est["Estoque Inicial (UC)"] = est.groupby("Família Comercial")["Estoque Final (UC)"].shift(1).fillna(0.0)

            # ===== Layout WIDE: meses como colunas; métricas como sub-colunas =====
            # Mapear nome do mês e aplicar escala
            est["Mês"] = est["mes"].map(MONTHS_PT)
            display_cols = ["Estoque Inicial (UC)","Produção (UC)","Vendas (UC)","Estoque Final (UC)"]
            est_disp = est.copy()
            for c in display_cols:
                est_disp[c] = (est_disp[c] / scale).round(0)

            # Pivot: index = Família; columns = Mês (top-level) × Métrica (second-level)
            base = est_disp[["Família Comercial","Mês"] + display_cols].set_index(["Família Comercial","Mês"])
            wide = base[display_cols].unstack("Mês")  # columns level0=metric, level1=month
            wide = wide.swaplevel(0, 1, axis=1)       # level0=month, level1=metric
            # Ordenar meses pelo número original (não exibido), respeitando apenas os presentes
            meses_presentes = est.sort_values("mes")["Mês"].dropna().unique().tolist()
            # Reindex suave (apenas meses presentes), mantendo ordem por ocorrência
            if len(meses_presentes) > 0:
                # construir MultiIndex na ordem desejada
                new_cols = []
                for m in meses_presentes:
                    for met in display_cols:
                        tup = (m, met)
                        if tup in wide.columns:
                            new_cols.append(tup)
                if new_cols:
                    wide = wide.reindex(columns=pd.MultiIndex.from_tuples(new_cols, names=wide.columns.names))

            # Render: DataFrame largo com rolagem horizontal
            st.dataframe(wide, use_container_width=True, hide_index=False)

            # Diagnóstico leve se tudo zerado
            try:
                if (est_disp[["Produção (UC)","Vendas (UC)"]].sum().sum() == 0):
                    st.warning("Rolagem calculada, porém UI/RE estão zerados para o recorte atual (ano/cutoff/escala). Verifique RES/RE.")
            except Exception:
                pass

            st.caption("Regra temporária: Estoque Inicial encadeado (mês a mês); Produção=UI; Vendas=UI (se Projeção) ou RE (se desmarcado). Cabeçalho agrupado por mês.")
    except Exception as e:
        st.error(f"[Rolagem de Estoque] Erro: {e}")


# ===============================
# Logs leves (INFO)
# ===============================
try:
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info("[POH] ano=%s cutoff=%s linhas(cap_vs_dem)=%s tec=%s fam=%s",
                 ano, cutoff,
                 cap_vs_dem.shape[0] if isinstance(cap_vs_dem, pd.DataFrame) else -1,
                 fam_tec['Tecnologia'].nunique() if isinstance(fam_tec, pd.DataFrame) and 'Tecnologia' in fam_tec.columns else 0,
                 fam_tec['Família Comercial'].nunique() if isinstance(fam_tec, pd.DataFrame) and 'Família Comercial' in fam_tec.columns else 0)
except Exception:
    pass
