# simulador_dre_streamlit/etl/transformers.py
# -*- coding: utf-8 -*-
"""
TRANSFORMERS — Regras de negócio (puras, sem I/O)

Objetivo:
- Padronizar e enriquecer as bases de realizado (CURRENT) e projeção (RES) para o simulador de DRE.
- Entregar TODAS as linhas do P&L vindas no realizado (não apenas 3-4 métricas).

Principais entregas:
- transform_realizado(df): normaliza dimensões, captura todas as métricas do DRE, faz parsing numérico BR,
  infere 'ano' pelo cenário quando necessário, remove TOTAL/SUBTOTAL e agrega por chaves.
- transform_projecao(df_wide): normaliza a base RES em formato wide (YYYY-MM no cabeçalho), gera ano/mês,
  coerção numérica e filtra TOTAL/SUBTOTAL se houver.

Dependências:
- utilitários expostos em etl/common.py: find_column, normalize_mes, etc.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence, Dict, List

import numpy as np
import pandas as pd

# Utils do projeto
from .common import (
    find_column,
    normalize_mes,
    # >>> INCLUSÃO: enriquecimento de Tecnologia (por SKU / fallback Família)
    apply_tecnologia,
)

# ============================================================
# Helpers de normalização e parsing
# ============================================================

def _strip_accents_lower(s: str) -> str:
    """minúsculas + sem acento (para matching/filtros)."""
    if s is None:
        return ""
    s2 = str(s).lower()
    rep = (
        ("á", "a"), ("à", "a"), ("â", "a"), ("ã", "a"),
        ("é", "e"), ("ê", "e"),
        ("í", "i"),
        ("ó", "o"), ("ô", "o"), ("õ", "o"),
        ("ú", "u"),
        ("ç", "c"),
    )
    for a, b in rep:
        s2 = s2.replace(a, b)
    return s2


def _normkey(s: str) -> str:
    """Normaliza rótulos de coluna para uma chave comparável (acentos, pontuação, espaços)."""
    s = _strip_accents_lower(s)
    s = (
        s.replace("_", " ")
         .replace("-", " ")
         .replace("/", " ")
         .replace(".", " ")
         .replace("  ", " ")
         .strip()
    )
    return s


def _to_number_br(x) -> float:
    """
    Converte entradas em formato BR para número.
    Regras:
      - Se já for numérico (int/float/np.number), retorna como float (preserva).
      - Se string com ',' e '.', assume '.' milhar e ',' decimal -> remove '.' e troca ',' por '.'
      - Se string só com ',', troca ',' por '.'
      - Se só com '.', usa como decimal padrão.
      - Remove 'R$', espaços; trata (1.234,56) como negativo.
      - Inválidos -> 0.0
    """
    import numpy as _np
    import pandas as _pd

    if isinstance(x, (int, float, _np.number)):
        try:
            if _np.isnan(x):  # type: ignore
                return 0.0
        except Exception:
            pass
        return float(x)

    if x is None:
        return 0.0

    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return 0.0

    neg = s.startswith("(") and s.endswith(")")
    if neg:
        s = s[1:-1].strip()

    s = s.replace("R$", "").replace("r$", "").replace(" ", "")

    has_comma = "," in s
    has_dot = "." in s

    if has_comma and has_dot:
        s = s.replace(".", "").replace(",", ".")
    elif has_comma and not has_dot:
        s = s.replace(",", ".")
    else:
        # só ponto ou dígitos -> segue como está
        pass

    try:
        v = float(s)
        return -v if neg else v
    except Exception:
        try:
            v = _pd.to_numeric(s, errors="coerce")
            v = 0.0 if _np.isnan(v) else float(v)
            return -v if neg else v
        except Exception:
            return 0.0


def _coerce_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Converte colunas para float com tolerância a formato BR. Ausentes viram 0.0."""
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].map(_to_number_br)
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
    return df


def _normalize_scenario(df: pd.DataFrame) -> pd.DataFrame:
    """Renomeia variações para 'cenario' quando existir uma coluna equivalente."""
    df = df.copy()
    scen = find_column(df, ["cenario", "Cenário", "Cenario", "cenário"])
    if scen and scen != "cenario":
        df = df.rename(columns={scen: "cenario"})
    return df


def _infer_year_from_scenario(s: str) -> Optional[int]:
    """Infere ano a partir de padrões comuns no texto do cenário."""
    if not s:
        return None
    ss = str(s)
    m = re.search(r"(20\d{2})", ss)  # BP2025
    if m:
        return int(m.group(1))
    m2 = re.search(r"(\d{2})[./](\d{2})", ss)  # 05.25 -> 2025
    if m2:
        yy = int(m2.group(2))
        return 2000 + yy
    m3 = re.search(r"(\d{2})\s*$", ss)  # ... 25
    if m3:
        yy = int(m3.group(1))
        if 20 <= yy <= 39:
            return 2000 + yy
    return None


# ============================================================
# Mapeamento de colunas do P&L (aliases → canônicos)
# ============================================================

_PNL_ALIASES: Dict[str, str] = {
    # dimensões
    "cenario": "cenario",
    "sistema": "sistema",
    "mes": "mes",
    "cod produto": "SKU",
    "cod. produto": "SKU",
    "codigo do produto": "SKU",
    "codigo produto": "SKU",
    "prod descricao": "produto_descricao",
    "produto descricao": "produto_descricao",
    "familia comercial": "Família Comercial",  # mantemos o acento como coluna canônica

    # volumes
    "volume (uc)": "volume_uc",
    "volume uc": "volume_uc",
    "volume sell out (uc)": "volume_sellout",
    "volume sellout (uc)": "volume_sellout",
    "volume sell out": "volume_sellout",
    "sell out": "volume_sellout",

    # receitas/margens/custos
    "receita bruta": "receita_bruta",
    "receita liquida": "receita_liquida",
    "convenio impostos": "convenio_impostos",
    "charge back": "charge_back",
    "insumos": "insumos",
    "resultado operacional": "resultado_operacional",
    "custos toll packer": "custos_toll_packer",
    "fretes t1": "fretes_t1",
    "estrutura industrial variavel": "estrutura_industrial_variavel",
    "custos variaveis": "custos_variaveis",
    "margem variavel": "margem_variavel",
    "estrutura industrial fixa": "estrutura_industrial_fixa",
    "margem bruta": "margem_bruta",
    "custos log t1 reg": "custos_log_t1_reg",
    "despesas secundarias (t2)": "despesas_secundarias_t2",
    "perdas": "perdas",
    "margem contribuicao": "margem_contribuicao",
    "dme": "dme",
    "margem contribuicao liquida": "margem_contribuicao_liquida",
    "opex leao reg": "opex_leao_reg",
    "depreciacao": "depreciacao",
    "ebitda": "ebitda",
}

# Lista completa de métricas que vamos coerçar/agregar se existirem na fonte
_PNL_METRICS = [
    "volume_uc", "volume_sellout",
    "receita_bruta", "receita_liquida", "convenio_impostos", "charge_back", "insumos",
    "resultado_operacional",
    "custos_toll_packer", "fretes_t1", "estrutura_industrial_variavel", "custos_variaveis",
    "margem_variavel", "estrutura_industrial_fixa", "margem_bruta",
    "custos_log_t1_reg", "despesas_secundarias_t2", "perdas",
    "margem_contribuicao", "dme", "margem_contribuicao_liquida",
    "opex_leao_reg", "depreciacao", "ebitda",
]


def _map_realizado_columns(cols: List[str]) -> Dict[str, str]:
    """Gera um rename_map para colunas do realizado com base em aliases/heurísticas."""
    rename: Dict[str, str] = {}
    for c in cols:
        k = _normkey(c)

        # Heurísticas específicas
        if k.startswith("cod ") and "produto" in k:             # "Cód. Produto"
            rename[c] = "SKU"; continue
        if k.startswith("prod ") and "descricao" in k:          # "Prod. Descrição"
            rename[c] = "produto_descricao"; continue
        if k.startswith("familia") and "comercial" in k:
            rename[c] = "Família Comercial"; continue
        if "volume" in k and "sell" in k:
            rename[c] = "volume_sellout"; continue

        # Aliases diretos
        if k in _PNL_ALIASES:
            rename[c] = _PNL_ALIASES[k]; continue

        # Variações de espaços redundantes
        k2 = " ".join(k.split())
        if k2 in _PNL_ALIASES:
            rename[c] = _PNL_ALIASES[k2]; continue
    return rename


# ============================================================
# Filtros de higiene — remover TOTAL/SUBTOTAL e linhas ruins
# ============================================================

_TOTAL_REGEX = re.compile(r"\b(total|sub[\s\-]?total|totalgeral|total geral|grand\s+total)\b", re.IGNORECASE)

def _drop_totals_subtotals_realizado(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regras:
      - Dropar linhas sem SKU (vazio/NA).
      - Dropar linhas que aparentem ser TOTAL/SUBTOTAL via colunas descritivas.
      - Dropar linhas com 'cenario' nulo E sem SKU.
    """
    df = df.copy()

    # SKU obrigatório
    sku_col = "SKU" if "SKU" in df.columns else None
    if sku_col:
        sku_str = df[sku_col].astype(str).str.strip()
        cond_no_sku = sku_str.eq("").fillna(True) | sku_str.str.lower().isin({"nan", "none", "null"})
    else:
        cond_no_sku = pd.Series([False] * len(df))

    # Detecta TOTAL/SUBTOTAL em campos descritivos
    cond_tot = pd.Series([False] * len(df))
    for c in ["produto_descricao", "Família Comercial", "sistema", "Sistema"]:
        if c in df.columns:
            norm = df[c].astype(str).map(_strip_accents_lower)
            cond_tot = cond_tot | norm.str.contains(_TOTAL_REGEX, na=False)

    # Cenário nulo
    scen_col = "cenario" if "cenario" in df.columns else None
    if scen_col:
        scen_str = df[scen_col].astype(str).str.strip()
        cond_nan_cen = scen_str.eq("").fillna(True) | scen_str.str.lower().isin({"nan", "none", "null"})
    else:
        cond_nan_cen = pd.Series([False] * len(df))

    drop_mask = cond_no_sku | cond_tot | (cond_no_sku & cond_nan_cen)
    return df[~drop_mask].copy()


def _drop_totals_subtotals_res(df: pd.DataFrame) -> pd.DataFrame:
    """Para o RES, remove linhas cuja família sugira TOTAL/SUBTOTAL."""
    df = df.copy()
    fam_col = "Família Comercial" if "Família Comercial" in df.columns else find_column(df, ["Familia Comercial"])
    if not fam_col:
        return df
    fam_norm = df[fam_col].astype(str).map(_strip_accents_lower)
    mask = fam_norm.str.contains(_TOTAL_REGEX, na=False)
    return df[~mask].copy()


# ============================================================
# TRANSFORMERS principais
# ============================================================

def transform_realizado(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realizado/Cenários → retorna dims + TODAS as métricas do P&L presentes na fonte.
    Passos:
      - Renomeia colunas por aliases/heurísticas (dims e métricas).
      - Normaliza 'cenario' e 'mes'; infere 'ano' pelo cenário se ausente.
      - Coerção numérica BR em todas as métricas conhecidas que existirem.
      - Remove TOTAL/SUBTOTAL e linhas sem SKU.
      - Enriquecimento: adiciona 'Tecnologia' (SKU absoluto; fallback Família).
      - Agrega por chaves (SKU, Família, Tecnologia, produto, sistema, cenario, ano, mes).
    """
    df = df.copy()

    # 1) Renomear colunas
    rename = _map_realizado_columns(list(df.columns))
    if rename:
        df = df.rename(columns=rename)

    # 2) Normalizar chaves
    df = _normalize_scenario(df)
    mes_col = find_column(df, ["mes", "Mês", "Mes"])
    if mes_col and mes_col != "mes":
        df = df.rename(columns={mes_col: "mes"})
    if "mes" in df.columns:
        if not np.issubdtype(df["mes"].dtype, np.number):
            df["mes"] = df["mes"].map(normalize_mes)
        df["mes"] = pd.to_numeric(df["mes"], errors="coerce").astype("Int64")

    # 3) Inferir ano, se faltar
    if "ano" not in df.columns or df["ano"].isna().all():
        scen_col = "cenario" if "cenario" in df.columns else find_column(df, ["Cenário", "cenario"])
        if scen_col:
            df["ano"] = df[scen_col].map(_infer_year_from_scenario).astype("Int64")

    # 4) Coerção numérica nas métricas do DRE que existirem
    to_coerce = [c for c in _PNL_METRICS if c in df.columns]
    if to_coerce:
        df = _coerce_numeric(df, to_coerce)

    # 5) Higiene
    df = _drop_totals_subtotals_realizado(df)

    # 6) Enriquecimento — Tecnologia (por SKU; fallback por Família)
    #    * chamada sem logger/base_dir explícitos para manter pureza da assinatura
    df = apply_tecnologia(df)

    # 7) Agregação por chaves (inclui Tecnologia se existir)
    dims = [c for c in ["SKU", "Família Comercial", "Tecnologia", "produto_descricao", "sistema", "cenario", "ano", "mes"] if c in df.columns]
    metrics = [c for c in _PNL_METRICS if c in df.columns]
    if dims and metrics:
        df = df.groupby(dims, dropna=False)[metrics].sum().reset_index()

    # 8) Ordenação de colunas: dims → métricas → demais
    ordered = dims + metrics + [c for c in df.columns if c not in dims + metrics]
    df = df[ordered]
    return df


def transform_projecao(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Projeções (RES working) em formato wide:
      - strip em nomes das colunas (havia espaços à esquerda);
      - identifica colunas YYYY-MM (ou YYYY/MM) no cabeçalho;
      - melt → extrai ano/mes diretamente do cabeçalho;
      - coerção numérica de 'volume';
      - padroniza 'Família Comercial';
      - remove TOTAL/SUBTOTAL se houver;
      - ENRIQUECE com 'Tecnologia' via fallback por Família.
    Saída: ['Família Comercial', 'Tecnologia', 'ano', 'mes', 'volume']
    """
    df = df_wide.copy()

    # 1) strip nos rótulos
    df.columns = [str(c).strip() for c in df.columns]

    # 2) detectar colunas de período
    month_cols = [c for c in df.columns if re.fullmatch(r"\d{4}[-/]\d{2}", str(c).strip())]
    if not month_cols:
        month_cols = [c for c in df.columns if re.search(r"\d{4}[-/]\d{2}", str(c))]

    # 3) melt
    id_vars = [c for c in df.columns if c not in month_cols]
    long = df.melt(id_vars=id_vars, value_vars=month_cols, var_name="periodo", value_name="volume")

    # 4) extrair ano/mes do cabeçalho
    def _extract_period(s: str) -> Optional[str]:
        m = re.search(r"(\d{4})[-/](\d{2})", str(s))
        return m.group(0) if m else None

    long["periodo"] = long["periodo"].map(_extract_period)
    long[["ano", "mes"]] = long["periodo"].str.split(r"[-/]", expand=True)
    long["ano"] = pd.to_numeric(long["ano"], errors="coerce").astype("Int64")
    long["mes"] = pd.to_numeric(long["mes"], errors="coerce").astype("Int64")
    long = long.drop(columns=["periodo"])

    # 5) coerção do volume
    long["volume"] = pd.to_numeric(long["volume"], errors="coerce").fillna(0.0).astype(float)

    # 6) família canônica
    fam_col = find_column(long, ["Família Comercial", "Familia Comercial"])
    if fam_col and fam_col != "Família Comercial":
        long = long.rename(columns={fam_col: "Família Comercial"})

    # 7) higiene RES
    long = _drop_totals_subtotals_res(long)

    # 8) Enriquecimento — Tecnologia (fallback por Família; RES não tem SKU)
    long = apply_tecnologia(long)

    # 9) seleção final (inclui Tecnologia)
    for c in ["Família Comercial", "Tecnologia", "ano", "mes", "volume"]:
        if c not in long.columns:
            long[c] = pd.Series([None] * len(long)) if c not in ("volume",) else 0.0

    return long[["Família Comercial", "Tecnologia", "ano", "mes", "volume"]]
