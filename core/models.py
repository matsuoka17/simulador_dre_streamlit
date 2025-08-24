# core/models.py
# -*- coding: utf-8 -*-
"""
Camada de MODELS (data access + consultas P&L).

Responsabilidades:
- Carregar os parquets ancorados em core.paths (SEM caminhos absolutos).
- Normalizar o CURRENT em formato LONG (['indicador_id','valor'] + dims).
- Descobrir cenários (inclui RE mais recente no padrão 'RE MM.AA').
- Entregar consultas P&L: long e matrizes (total e por família/tecnologia).
- Expor constantes e utilitários compartilhados (ex.: conv/dme tax, meses-PT).

Regras/Fórmulas (referência do projeto):
    conv_tax = -0.265
    dme_tax  = -0.102
    Linhas âncora em negrito: vol, rb, rl, cv, mv, mb, mc, mcl, ebitda
    Destaque amarelo: ro (feito na UI)

BP/RE/Realizado YTD: vêm de current.parquet (já com as linhas do P&L).
Realizado YTG: calculado na camada de cálculo (core.calculator*), usando:
    - volumes do res_working.parquet (UI/RES),
    - parâmetros de preço/custos (data/premissas_pnl/base_calculos.xlsx),
    - EI fixo/variável e T1/T2/Perdas/Opex/CB/Depre copiados do RE.

Obs.: Aqui em models centralizamos a leitura/normalização e as matrizes;
      o cálculo YTG em si continua no motor (calculator), mas expomos
      utilitários e séries que o motor pode reutilizar.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import re

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------#
# Helpers de normalização (slug/canonização/volume dedup)
# -----------------------------------------------------------------------------#
import unicodedata
import re as _re

def _slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or "").strip())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = _re.sub(r"[^a-z0-9]+", "_", s.lower())
    return _re.sub(r"_+", "_", s).strip("_")

# Mapa de sinônimos → IDs canônicos do P&L
CANON_MAP = {
    # volumes
    "volume": "volume_uc",
    "volume_uc": "volume_uc",
    "volume_sellout": "volume_uc",
    "sellout": "volume_uc",
    # RB / RL / Convênio
    "receita_bruta": "receita_bruta",
    "rb": "receita_bruta",
    "convenio_impostos": "convenio_impostos",
    "convenio_imposto": "convenio_impostos",
    "convenio_impostos_pis_cofins_icms": "convenio_impostos",
    "receita_liquida": "receita_liquida",
    "rl": "receita_liquida",
    # custos variáveis
    "insumos": "insumos",
    "custo_insumo": "insumos",
    "custos_toll_packer": "custos_toll_packer",
    "toll": "custos_toll_packer",
    "fretes_t1": "fretes_t1",
    "frete_t1": "fretes_t1",
    "t1_frete": "fretes_t1",
    "estrutura_industrial_variavel": "estrutura_industrial_variavel",
    "ei_var": "estrutura_industrial_variavel",
    "custos_variaveis": "custos_variaveis",
    "margem_variavel": "margem_variavel",
    "estrutura_industrial_fixa": "estrutura_industrial_fixa",
    "ei_fixa": "estrutura_industrial_fixa",
    "margem_bruta": "margem_bruta",
    # despesas logísticas/operacionais
    "custos_log_t1_reg": "custos_log_t1_reg",
    "t1_reg": "custos_log_t1_reg",
    "despesas_secundarias_t2": "despesas_secundarias_t2",
    "t2": "despesas_secundarias_t2",
    "perdas": "perdas",
    "dme": "dme",
    "margem_contribuicao": "margem_contribuicao",
    "mcl": "margem_contribuicao_liquida",
    "margem_contribuicao_liquida": "margem_contribuicao_liquida",
    "opex": "opex_leao_reg",
    "opex_leao_reg": "opex_leao_reg",
    "chargeback": "chargeback",
    "charge_back": "chargeback",
    "cb": "chargeback",
    "resultado_operacional": "resultado_operacional",
    "depreciacao": "depreciacao",
    "dep": "depreciacao",
    "ebitda": "ebitda",
}

def _canonicalize_indicador(series: pd.Series) -> pd.Series:
    return series.astype(str).map(lambda x: CANON_MAP.get(_slug(x), x))

# -----------------------------------------------------------------------------#
# Paths oficiais do projeto
# -----------------------------------------------------------------------------#
from core.paths import (
    PROJECT_ROOT, DATA_DIR, PARQUET_DIR,
    CURRENT_PARQUET, RES_WORKING_PARQUET,
    PREMISSAS_PNL_DIR, BASE_CALCULOS_XLSX,
)

# -----------------------------------------------------------------------------#
# Logging
# -----------------------------------------------------------------------------#
logger = logging.getLogger("simulador.models")
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------#
# Constantes de projeto
# -----------------------------------------------------------------------------#
CONV_TAX: float = -0.265  # Convênio/Impostos = CONV_TAX * Receita Bruta
DME_TAX: float = -0.102   # DME               = DME_TAX  * Receita Líquida

def get_conv_tax() -> float: return CONV_TAX
def get_dme_tax() -> float:  return DME_TAX

MONTH_MAP_PT = {
    1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez",
}
MONTHS_PT = [MONTH_MAP_PT[i] for i in range(1, 13)]

# Ordem preferida das contas do P&L (para matrizes)
PNL_PREF_ORDER = [
    "volume_uc",
    "receita_bruta", "convenio_impostos", "receita_liquida",
    "insumos", "custos_toll_packer", "fretes_t1",
    "estrutura_industrial_variavel", "custos_variaveis", "margem_variavel",
    "estrutura_industrial_fixa", "margem_bruta",
    "custos_log_t1_reg", "despesas_secundarias_t2", "perdas",
    "margem_contribuicao", "dme", "margem_contribuicao_liquida",
    "opex_leao_reg", "chargeback",
    "resultado_operacional", "depreciacao", "ebitda",
]

# -----------------------------------------------------------------------------#
# Helpers internos
# -----------------------------------------------------------------------------#
def _file_signature(p: Path) -> str:
    """Assinatura robusta para chave de cache; tolera arquivo ausente."""
    p = Path(p)
    try:
        stt = p.stat()
        return f"{p.as_posix()}|{stt.st_mtime_ns}|{stt.st_size}"
    except FileNotFoundError:
        return f"{p.as_posix()}|MISSING"

def _fam_col(df: pd.DataFrame) -> str:
    for c in ("Família Comercial", "Familia Comercial", "familia_comercial", "Familia"):
        if c in df.columns:
            return c
    return "Família Comercial"

def _ensure_ano_mes(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "ano" in d.columns:
        d["ano"] = pd.to_numeric(d["ano"], errors="coerce").astype("Int64")
    if "mes" in d.columns:
        d["mes"] = pd.to_numeric(d["mes"], errors="coerce").astype("Int64")
    return d

def _order_metrics(metrics: List[str]) -> List[str]:
    pref = [m for m in PNL_PREF_ORDER if m in metrics]
    rest = [m for m in metrics if m not in pref]
    return pref + sorted(rest)

def _scenario_mask(series: pd.Series, like: str) -> pd.Series:
    """Filtro robusto para cenários:
    - 'RE'  → ^RE\b  (prefixo RE)
    - 'BP'  → ^BP\b  (prefixo BP)
    - 'Realizado' → contém 'realiz'
    - outro texto → contains normal (case-insensitive)
    """
    if series is None:
        return pd.Series(False, index=pd.RangeIndex(0))
    like = (like or "").strip().lower()
    s = series.astype(str)

    if like in {"re", "r.e", "r e"}:
        return s.str.match(r"(?i)^\s*RE\b")
    if like in {"bp"}:
        return s.str.match(r"(?i)^\s*BP\b")
    if "realiz" in like:
        return s.str.contains(r"(?i)realiz")

    return s.str.contains(_re.escape(like), case=False, na=False)

# -----------------------------------------------------------------------------#
# Loaders cacheados (RAW) – CURRENT e RES_WORKING
# -----------------------------------------------------------------------------#
@st.cache_data(show_spinner=False)
def _load_current_cached(_sig: str) -> pd.DataFrame:
    if not CURRENT_PARQUET.exists():
        logger.warning("current.parquet não encontrado em %s", CURRENT_PARQUET)
        return pd.DataFrame()
    df = pd.read_parquet(CURRENT_PARQUET)
    logger.info("CURRENT carregado: %s (linhas=%d, cols=%d)", CURRENT_PARQUET, len(df), df.shape[1])
    return df

def load_current_raw() -> pd.DataFrame:
    return _load_current_cached(_file_signature(CURRENT_PARQUET))

@st.cache_data(show_spinner=False)
def _load_res_cached(_sig: str) -> pd.DataFrame:
    if not RES_WORKING_PARQUET.exists():
        logger.warning("res_working.parquet não encontrado em %s", RES_WORKING_PARQUET)
        return pd.DataFrame()
    df = pd.read_parquet(RES_WORKING_PARQUET)
    logger.info("RES_WORKING carregado: %s (linhas=%d, cols=%d)", RES_WORKING_PARQUET, len(df), df.shape[1])
    return df

def load_res_raw() -> pd.DataFrame:
    return _load_res_cached(_file_signature(RES_WORKING_PARQUET))

# -----------------------------------------------------------------------------#
# Normalização LONG (CURRENT)
# -----------------------------------------------------------------------------#
DIM_CANDIDATES = [
    "Família Comercial", "Familia Comercial", "Tecnologia",
    "SKU", "produto_descricao", "sistema",
    "cenario", "ano", "mes", "cliente", "canal",
]

def _normalize_current_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte CURRENT para long ['indicador_id','valor'] caso esteja em 'wide'.
    - Canoniza aliases (ex.: volume_sellout → volume_uc)
    - Dedup conservador (wide): prioriza 'volume'/'volume_uc' em relação a 'volume_sellout'
    """
    if df.empty:
        return pd.DataFrame(columns=["indicador_id", "valor"])

    df = _ensure_ano_mes(df)
    fam = _fam_col(df)
    if fam != "Família Comercial" and fam in df.columns:
        df = df.rename(columns={fam: "Família Comercial"})

    if {"indicador_id", "valor"}.issubset(df.columns):
        out = df.copy()
        out["indicador_id"] = _canonicalize_indicador(out["indicador_id"])
        out["valor"] = pd.to_numeric(out["valor"], errors="coerce").fillna(0.0)
        return out

    dims = [c for c in DIM_CANDIDATES if c in df.columns]
    metric_cols = [c for c in df.columns if c not in dims]
    metric_cols = [c for c in metric_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not metric_cols:
        logger.warning("CURRENT sem colunas numéricas para melt; retornando vazio.")
        return pd.DataFrame(columns=dims + ["indicador_id", "valor"])

    melted = df.melt(
        id_vars=dims, value_vars=metric_cols,
        var_name="orig_indicador", value_name="valor",
    )
    melted["valor"] = pd.to_numeric(melted["valor"], errors="coerce").fillna(0.0)
    melted["orig_indicador_slug"] = melted["orig_indicador"].map(_slug)
    melted["indicador_id"] = melted["orig_indicador_slug"].map(lambda s: CANON_MAP.get(s, s))

    vol_mask = melted["indicador_id"].eq("volume_uc")
    if vol_mask.any():
        rank_map = {"volume": 2, "volume_uc": 2, "sellout": 1, "volume_sellout": 1}
        vol = melted.loc[vol_mask].copy()
        vol["__rank__"] = vol["orig_indicador_slug"].map(rank_map).fillna(0).astype(int)
        grain = [c for c in dims] + ["indicador_id"]
        vol = vol.sort_values("__rank__", ascending=False).drop_duplicates(subset=grain, keep="first")
        others = melted.loc[~vol_mask].copy()
        melted = pd.concat([others, vol.drop(columns="__rank__", errors="ignore")], ignore_index=True)

    out = melted[dims + ["indicador_id", "valor"]].copy()
    return out

@st.cache_data(show_spinner=False)
def load_current_long(filter_cenario: Optional[str] = None) -> pd.DataFrame:
    """
    Retorna CURRENT em LONG com dims + ['indicador_id','valor'].
    Se `filter_cenario` for informado, aplica filtro robusto (sem confundir RE com Realizado).
    """
    raw = load_current_raw()
    d = _normalize_current_long(raw)
    if filter_cenario and "cenario" in d.columns:
        mask = _scenario_mask(d["cenario"], filter_cenario)
        d = d.loc[mask].copy()
    return d

# -----------------------------------------------------------------------------#
# NOVO: Normalização LONG (RES_WORKING) e volumes por família
# -----------------------------------------------------------------------------#
@st.cache_data(show_spinner=False)
def load_res_working_long() -> pd.DataFrame:
    """
    Lê `res_working.parquet` e devolve em LONG com dims + ['indicador_id','valor'].
    - Se já vier em long (indicador_id/valor), apenas higieniza.
    - Se vier wide, procura colunas de volume e derrete como 'volume_uc'.
    """
    df = load_res_raw()
    if df.empty:
        return pd.DataFrame(columns=["indicador_id", "valor"])

    df = _ensure_ano_mes(df)
    fam = _fam_col(df)
    if fam != "Família Comercial" and fam in df.columns:
        df = df.rename(columns={fam: "Família Comercial"})

    # Caso já esteja em long
    if {"indicador_id", "valor"}.issubset(df.columns):
        out = df.copy()
        out["indicador_id"] = _canonicalize_indicador(out["indicador_id"])
        out["valor"] = pd.to_numeric(out["valor"], errors="coerce").fillna(0.0)
        return out

    # Caso wide: detecta coluna(es) de volume
    dims = [c for c in DIM_CANDIDATES if c in df.columns]
    cand_vol = []
    for c in df.columns:
        cl = str(c).strip().lower()
        if c in dims:
            continue
        if cl in {"volume", "volume_uc", "qtd", "quantidade", "uc"}:
            cand_vol.append(c)

    if not cand_vol:
        # Tenta "valor" como volume
        for c in df.columns:
            if str(c).strip().lower() == "valor" and c not in dims:
                cand_vol.append(c)
                break

    if not cand_vol:
        # Não conseguiu detectar métricas: retorna vazio com contrato.
        return pd.DataFrame(columns=dims + ["indicador_id", "valor"])

    melted = df.melt(
        id_vars=dims, value_vars=cand_vol,
        var_name="orig_indicador", value_name="valor",
    )
    melted["valor"] = pd.to_numeric(melted["valor"], errors="coerce").fillna(0.0)
    melted["indicador_id"] = "volume_uc"  # baseline RES é volume
    out = melted[dims + ["indicador_id", "valor"]].copy()
    return out

@st.cache_data(show_spinner=False)
def res_volume_by_family_long() -> pd.DataFrame:
    """
    Volumes por família/ano/mês a partir do RES_WORKING.
    Retorna: ['Família Comercial','ano','mes','volume'].
    """
    long = load_res_working_long()
    if long.empty:
        return pd.DataFrame(columns=["Família Comercial", "ano", "mes", "volume"])
    sub = long[long["indicador_id"] == "volume_uc"].copy()
    if sub.empty:
        return pd.DataFrame(columns=["Família Comercial", "ano", "mes", "volume"])
    out = (
        sub.groupby(["Família Comercial", "ano", "mes"], as_index=False, dropna=False)["valor"].sum()
           .rename(columns={"valor": "volume"})
    )
    out["ano"] = pd.to_numeric(out["ano"], errors="coerce").astype("Int64")
    out["mes"] = pd.to_numeric(out["mes"], errors="coerce").astype("Int64")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0).astype(float)
    out["Família Comercial"] = out["Família Comercial"].astype(str).str.strip()
    return out

# -----------------------------------------------------------------------------#
# Descoberta de cenários / anos / RE mais recente (CURRENT)
# -----------------------------------------------------------------------------#
def list_scenarios_current() -> List[str]:
    df = load_current_long()
    if "cenario" not in df.columns or df.empty:
        return []
    sc = df["cenario"].dropna().astype(str).str.strip().unique().tolist()
    return sorted(sc)

def years_available_current() -> List[int]:
    df = load_current_long()
    if "ano" not in df.columns or df.empty:
        return []
    anos = pd.to_numeric(df["ano"], errors="coerce").dropna().astype(int).unique().tolist()
    return sorted(anos)

_RE_MMYY_RE = re.compile(r"(?i)\bRE\s*([0-1]?\d)[\.\-\/]?\s*(\d{2})\b")  # captura mm e aa

def find_latest_re_scenario(year: Optional[int] = None) -> Optional[str]:
    df = load_current_long()
    if "cenario" not in df.columns or df.empty:
        return None
    unique = df["cenario"].dropna().astype(str).unique().tolist()

    matches: List[Tuple[str, int, int]] = []  # (cenário, aa, mm)
    for s in unique:
        m = _RE_MMYY_RE.search(s)
        if not m:
            continue
        mm = int(m.group(1)); aa = int(m.group(2))
        if 1 <= mm <= 12:
            matches.append((s, aa, mm))

    if not matches:
        cand = [s for s in unique if s.strip().upper().startswith("RE")]
        return sorted(cand)[-1] if cand else None

    if year is not None:
        aa_target = year % 100
        same_year = [t for t in matches if t[1] == aa_target]
        if same_year:
            s, _, _ = max(same_year, key=lambda t: (t[1], t[2]))
            logger.info("RE mais recente para %s: %s", year, s)
            return s

    s, _, _ = max(matches, key=lambda t: (t[1], t[2]))
    logger.info("RE mais recente (geral): %s", s)
    return s

def scenario_realizado_for_year(year: int) -> Optional[str]:
    df = load_current_long()
    if df.empty or "cenario" not in df.columns: return None
    mask = df["cenario"].astype(str).str.contains("realiz", case=False, na=False)
    cand = df.loc[mask & (df["ano"] == year), "cenario"].dropna().astype(str).unique().tolist()
    return cand[0] if cand else None

def scenarios_for_year(year: int) -> List[str]:
    df = load_current_long()
    if df.empty: return []
    cand = df.loc[df["ano"] == year, "cenario"].dropna().astype(str).unique().tolist()

    def _rank(s: str) -> Tuple[int, str]:
        up = s.upper().replace(" ", "")
        if up.startswith("BP"): return (0, s)
        if up.startswith("RE"): return (1, s)
        if "REALIZ" in up:     return (2, s)
        return (3, s)

    return sorted(cand, key=_rank)

def realized_cutoff_by_year(cenario_like: str = "Realizado") -> Dict[int, int]:
    """
    Retorna {ano: último mês com volume_uc > 0} para o cenário que contém `cenario_like`.
    """
    d = load_current_long(filter_cenario=cenario_like)
    if d.empty:
        return {}
    sub = d[d["indicador_id"] == "volume_uc"]
    g = sub.groupby(["ano", "mes"], dropna=False)["valor"].sum().reset_index()
    out: Dict[int, int] = {}
    for y, part in g.groupby("ano"):
        pos = part.loc[part["valor"] > 0, "mes"]
        out[int(y)] = int(pos.max()) if not pos.empty else 0
    return out

def res_volume_total_by_month(year: int) -> Dict[int, float]:
    """
    Retorna volumes mensais totais do RES_WORKING para um ano.
    """
    df = load_res_raw()
    if df.empty or "ano" not in df.columns:
        return {}
    # Detectar coluna de volume
    vol_col = None
    for c in df.columns:
        if str(c).lower() in {"volume", "volume_uc", "qtd", "quantidade", "uc", "valor"}:
            vol_col = c
            break
    if vol_col is None:
        return {}
    sub = df[df["ano"] == year]
    if sub.empty:
        return {}
    g = sub.groupby("mes")[vol_col].sum()
    return {int(m): float(g.get(m, 0.0)) for m in range(1, 13)}

# -----------------------------------------------------------------------------#
# DRE LONG e Matrizes (CURRENT)
# -----------------------------------------------------------------------------#
def dre_long(
    cenario_like: Optional[str] = None,
    year: Optional[int] = None,
    keep_dims: List[str] = (
        "Família Comercial", "Tecnologia", "SKU",
        "produto_descricao", "sistema", "cliente", "canal",
        "cenario", "ano", "mes",
    ),
) -> pd.DataFrame:
    """
    Retorna CURRENT em formato LONG filtrado por cenário e/ou ano.
    Colunas: keep_dims ∩ df + ['indicador_id','valor'].
    """
    df = load_current_long(filter_cenario=cenario_like)
    if df.empty:
        return df
    if year is not None and "ano" in df.columns:
        df = df.loc[pd.to_numeric(df["ano"], errors="coerce") == int(year)].copy()
    have = [c for c in keep_dims if c in df.columns]
    out = df[have + ["indicador_id", "valor"]].copy()
    out["valor"] = pd.to_numeric(out["valor"], errors="coerce").fillna(0.0).astype(float)
    if "ano" in out.columns:
        out["ano"] = pd.to_numeric(out["ano"], errors="coerce").astype("Int64")
    if "mes" in out.columns:
        out["mes"] = pd.to_numeric(out["mes"], errors="coerce").astype("Int64")
    return out

def dre_matrix_total(year: int, cenario_like: Optional[str] = "Realizado") -> pd.DataFrame:
    """
    Retorna matriz P&L total (linhas = Conta, colunas = Jan..Dez + Total Ano) para um ano e um filtro de cenário.
    """
    df = dre_long(cenario_like=cenario_like, year=year)
    cols = ["Conta"] + MONTHS_PT + ["Total Ano"]
    if df.empty:
        return pd.DataFrame(columns=cols)

    g = df.groupby(["indicador_id", "mes"], dropna=False)["valor"].sum().reset_index()
    mat = (
        g.pivot_table(index="indicador_id", columns="mes", values="valor", aggfunc="sum")
        .fillna(0.0)
        .reindex(columns=list(range(1, 13)), fill_value=0.0)
    )
    mat.columns = [MONTH_MAP_PT[int(m)] for m in mat.columns]
    mat["Total Ano"] = mat.sum(axis=1)
    mat.index = mat.index.rename("Conta")
    mat = mat.reset_index()

    if "Conta" not in mat.columns:
        rename_map = {}
        if "indicador_id" in mat.columns: rename_map["indicador_id"] = "Conta"
        if "index" in mat.columns:        rename_map["index"] = "Conta"
        if rename_map:
            mat = mat.rename(columns=rename_map)
        if "Conta" not in mat.columns:
            mat.insert(0, "Conta", "DESCONHECIDO")

    try:
        ordered = _order_metrics(mat["Conta"].astype(str).tolist())
        mat = mat.set_index("Conta").reindex(ordered).reset_index()
    except Exception:
        if "Conta" in mat.columns:
            mat = mat.drop_duplicates(subset=["Conta"])

    for c in MONTHS_PT + ["Total Ano"]:
        if c in mat.columns:
            mat[c] = pd.to_numeric(mat[c], errors="coerce").fillna(0.0).round(0).astype(int)

    for c in cols:
        if c not in mat.columns:
            mat[c] = 0 if c != "Conta" else mat.get("Conta", "DESCONHECIDO")
    mat = mat[cols]
    return mat

def dre_matrix_by_family(
    year: int,
    cenario_like: Optional[str] = "Realizado",
    families: Optional[List[str]] = None,
    include_tecnologia: bool = False,
) -> pd.DataFrame:
    """
    Matriz por Família (e opcionalmente por Tecnologia):
      linhas = (Família [,Tecnologia], Conta), colunas = meses + 'Total Ano'
    """
    dims = ["Família Comercial", "indicador_id", "mes"]
    if include_tecnologia:
        dims.insert(1, "Tecnologia")

    df = dre_long(
        cenario_like=cenario_like, year=year,
        keep_dims=["Família Comercial", "Tecnologia", "ano", "mes", "indicador_id", "valor", "cenario"]
    )
    if df.empty:
        idx_cols = ["Família Comercial"] + (["Tecnologia"] if include_tecnologia else []) + ["Conta"]
        return pd.DataFrame(columns=idx_cols + MONTHS_PT + ["Total Ano"])

    if families:
        df = df[df["Família Comercial"].astype(str).isin(families)]

    g = df.groupby(dims, dropna=False)["valor"].sum().reset_index()
    idx_cols = ["Família Comercial"] + (["Tecnologia"] if include_tecnologia else []) + ["Conta"]
    wide = g.rename(columns={"indicador_id": "Conta"}).pivot_table(
        index=idx_cols, columns="mes", values="valor", aggfunc="sum"
    ).fillna(0.0)

    wide = wide.reindex(columns=list(range(1, 13)), fill_value=0.0)
    wide.columns = [MONTH_MAP_PT[m] for m in wide.columns]
    wide["Total Ano"] = wide.sum(axis=1)

    wide = wide.sort_index().reset_index()

    def _reindex_block(block: pd.DataFrame) -> pd.DataFrame:
        return (block.set_index("Conta")
                .reindex(_order_metrics(block["Conta"].tolist()))
                .reset_index())

    if include_tecnologia:
        parts = []
        for (fam, tec), part in wide.groupby(["Família Comercial", "Tecnologia"], sort=False):
            parts.append(_reindex_block(part).assign(**{"Família Comercial": fam, "Tecnologia": tec}))
        wide = pd.concat(parts, ignore_index=True)
    else:
        parts = []
        for fam, part in wide.groupby("Família Comercial", sort=False):
            parts.append(_reindex_block(part).assign(**{"Família Comercial": fam}))
        wide = pd.concat(parts, ignore_index=True)

    for c in MONTHS_PT + ["Total Ano"]:
        if c in wide.columns:
            wide[c] = pd.to_numeric(wide[c], errors="coerce").fillna(0.0).round(0).astype(int)
    return wide

# -----------------------------------------------------------------------------#
# Exposição de utilitários para o motor de cálculo (calculator)
# -----------------------------------------------------------------------------#
def get_series_from_scenario(indic_id: str, scenario: str, year: int) -> Dict[int, float]:
    """
    Série mensal total (empresa) para um indicador em `scenario`/`year`.
    Retorna {m: valor}, m=1..12.
    """
    d = dre_long(cenario_like=scenario, year=year, keep_dims=["ano", "mes", "indicador_id", "valor"])
    if d.empty:
        return {m: 0.0 for m in range(1, 13)}
    sub = d[d["indicador_id"] == indic_id]
    g = sub.groupby("mes")["valor"].sum()
    return {int(m): float(g.get(m, 0.0)) for m in range(1, 13)}

def get_series_by_family_from_scenario(indic_id: str, scenario: str, year: int) -> Dict[Tuple[str, int], float]:
    """
    Série mensal por família para um indicador em `scenario`/`year`.
    Retorna {(familia, m): valor}.
    """
    d = dre_long(cenario_like=scenario, year=year,
                 keep_dims=["Família Comercial", "ano", "mes", "indicador_id", "valor"])
    if d.empty:
        return {}
    sub = d[d["indicador_id"] == indic_id]
    g = sub.groupby(["Família Comercial", "mes"])["valor"].sum()
    return {(str(f), int(m)): float(v) for (f, m), v in g.items()}

# -----------------------------------------------------------------------------#
# Base de parâmetros (Excel) – para o motor de cálculo reutilizar
# -----------------------------------------------------------------------------#
@st.cache_data(show_spinner=False)
def load_base_calculos() -> pd.DataFrame:
    """
    Lê data/premissas_pnl/base_calculos.xlsx e normaliza colunas:
    'Indicador','Família Comercial','Mês','Valor','%'.
    """
    p = BASE_CALCULOS_XLSX
    if not p.exists():
        logger.warning("Base de cálculos não encontrada: %s", p)
        return pd.DataFrame(columns=["Indicador", "Família Comercial", "Mês", "Valor", "%"])
    df = pd.read_excel(p)
    rename = {}
    for c in df.columns:
        l = str(c).strip().lower()
        if l.startswith("indic"):
            rename[c] = "Indicador"
        elif l.startswith("fam"):
            rename[c] = "Família Comercial"
        elif l in {"mes", "mês", "meses"}:
            rename[c] = "Mês"
        elif l in {"valor", "val"}:
            rename[c] = "Valor"
        elif l in {"%", "pct", "percentual"}:
            rename[c] = "%"
    if rename:
        df = df.rename(columns=rename)
    logger.info("Base de cálculos carregada: %s (linhas=%d)", p, len(df))
    return df
