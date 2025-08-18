# -*- coding: utf-8 -*-
"""
ETL para gerar data/current.parquet a partir do Excel "largo" (realizado)
- Preserva literalmente o campo 'Cenário' do Excel:
  cenario_label = Cenário (texto original, strip)
  cenario       = cenario_label  (sem normalização)

- Remove linhas de subtotal/ALL
- Converte "largo" -> "longo"
- Coerção de mês "Jan/Fev..." -> 1..12 e ano
- Mapeia indicadores para IDs canônicos
- (Opcional) Aplica de-para de Família Comercial (etl/familia/current.xlsx)
- Salva em data/current.parquet + data/archive/{timestamp}_current.parquet
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("etl")

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../simulador_dre_streamlit
ETL_DIR      = PROJECT_ROOT / "etl"
REALIZADO_RAW = ETL_DIR / "realizado" / "raw"
FAMILIA_FILE  = ETL_DIR / "familia" / "current.xlsx"
PARQUET_OUT   = PROJECT_ROOT / "data" / "current.parquet"
ARCHIVE_DIR   = PROJECT_ROOT / "data" / "archive"

# -----------------------------------------------------------------------------
# Configurações/mapeamentos
# -----------------------------------------------------------------------------
# Colunas "meta" esperadas no Excel largo (se não existirem todas, seguimos com as que houver)
META_COLS = ["Cenário", "Sistema", "Mês", "Cód. Produto", "Prod. Descrição"]

# Mapa de indicadores (nome na planilha -> id canônico)
INDICATOR_MAP: Dict[str, str] = {
    "Volume Sell Out (UC)": "volume_sellout",
    "Volume (UC)": "volume_uc",
    "Receita Bruta": "receita_bruta",
    "Convênio/Impostos": "convenio_impostos",
    "Receita Líquida": "receita_liquida",
    "Insumos": "insumos",
    "Custos Toll Packer": "custos_toll_packer",
    "Concentrado/Incentivo de Vendas": "concentrado_incentivos",
    "Fretes T1": "fretes_t1",
    "Estrutura Industrial - Variável": "estrutura_industrial_variavel",
    "Custos Variáveis": "custos_variaveis",
    "Margem Variável": "margem_variavel",
    "Estrutura Industrial - Fixa": "estrutura_industrial_fixa",
    "Estrutura Industrial - Fixa Reg": "estrutura_industrial_fixa",
    "Margem Bruta": "margem_bruta",
    "Custos Log. T1 Reg": "custos_log_t1_reg",
    "Despesas Secundárias (T2)": "despesas_secundarias_t2",
    "Perdas": "perdas",
    "Margem Contribuição": "margem_contribuicao",
    "DME": "dme",
    "Margem Contribuição Líquida": "margem_contribuicao_liquida",
    "Ajuste Cota Fixa": "ajuste_cota_fixa",
    "Opex Leão Reg": "opex_leao_reg",
    "Cluster CCIL": "cluster_ccil",
    "Charge Back": "chargeback",
    "R. Oper. antes Regionalização": "resultado_antes_regionalizacao",
    "Custo Regionalizacao": "custo_regionalizacao",
    "Resultado Operacional": "resultado_operacional",
    "Depreciacao": "depreciacao",
    "EBITDA": "ebitda",
}

# Meses PT-BR (curtos e longos) -> número
MES_NAMES_PT = {
    "jan": 1, "janeiro": 1,
    "fev": 2, "fevereiro": 2,
    "mar": 3, "março": 3, "marco": 3,
    "abr": 4, "abril": 4,
    "mai": 5, "maio": 5,
    "jun": 6, "junho": 6,
    "jul": 7, "julho": 7,
    "ago": 8, "agosto": 8,
    "set": 9, "setembro": 9,
    "out": 10, "outubro": 10,
    "nov": 11, "novembro": 11,
    "dez": 12, "dezembro": 12,
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _find_latest_source(directory: Path) -> Path:
    exts = (".xlsx", ".xls", ".csv")
    files = [p for p in directory.glob("*") if p.suffix.lower() in exts and not p.name.startswith("~$")]
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo encontrado em {directory}")
    return max(files, key=lambda p: p.stat().st_mtime)

def _normalize_mes(v) -> Optional[int]:
    if pd.isna(v):
        return None
    if isinstance(v, (int, float)):
        i = int(v)
        return i if 1 <= i <= 12 else None
    s = str(v).strip()
    if not s:
        return None
    sl = s.lower()
    if sl in {"all", "total", "subtotal"}:
        return None
    if sl in MES_NAMES_PT:
        return MES_NAMES_PT[sl]
    if sl.isdigit():
        i = int(sl)
        return i if 1 <= i <= 12 else None
    # tenta AAAA-MM-DD
    m = pd.to_datetime(s, errors="coerce")
    if pd.notna(m):
        mm = int(m.month)
        return mm if 1 <= mm <= 12 else None
    return None

def _infer_ano_from_cenario_literal(cenario_label: str) -> Optional[int]:
    """Se não houver coluna 'Ano' explícita, tenta inferir a partir do texto literal do 'Cenário'."""
    if not isinstance(cenario_label, str):
        return None
    s = cenario_label.strip()
    # Procura AAAA
    m = pd.Series(s).str.extract(r"(\d{4})").iloc[0, 0]
    if pd.notna(m):
        return int(m)
    # Procura dois dígitos finais (ex.: Realizado 25 -> 2025; heurística)
    m2 = pd.Series(s).str.extract(r"(\d{2})\s*$").iloc[0, 0]
    if pd.notna(m2):
        yy = int(m2)
        return 2000 + yy if yy < 70 else 1900 + yy
    return None

def _drop_excel_subtotals(df: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas de subtotal/ALL no Excel largo (ex.: 'Valores', 'Total', 'ALL')."""
    if df.empty:
        return df
    cols = df.columns
    desc_cols = [c for c in ["Prod. Descrição", "Prod Descrição", "Produto", "Descrição", "Descricao"] if c in cols]
    code_cols = [c for c in ["Cód. Produto", "Cod. Produto", "Código", "Codigo", "SKU"] if c in cols]

    mask = pd.Series(False, index=df.index)
    if desc_cols:
        pattern = r"^\s*(valores?|total|subtotal|geral|all)\s*$"
        for c in desc_cols:
            mask |= df[c].astype(str).str.strip().str.lower().str.match(pattern, na=False)
    if code_cols:
        for c in code_cols:
            mask |= df[c].astype(str).str.strip().str.upper().isin(["ALL", "TOTAL"])

    removed = int(mask.sum())
    if removed > 0:
        log.info(f"Removendo {removed} linha(s) de subtotal/ALL.")
    return df.loc[~mask].copy()

# -----------------------------------------------------------------------------
# ETL
# -----------------------------------------------------------------------------
@dataclass
class ETLResult:
    df_long: pd.DataFrame
    source_path: Path

def load_realizado_largo() -> ETLResult:
    src = _find_latest_source(REALIZADO_RAW)
    log.info(f"Fonte 'realizado' (largo): {src}")
    if src.suffix.lower() == ".csv":
        df = pd.read_csv(src)
    else:
        df = pd.read_excel(src)
    df = _drop_excel_subtotals(df)
    return ETLResult(df_long=df, source_path=src)

def detect_indicator_columns(df: pd.DataFrame) -> List[str]:
    found = [c for c in INDICATOR_MAP.keys() if c in df.columns]
    if not found:
        raise ValueError("Nenhuma coluna de indicador conhecida foi encontrada no Excel.")
    return found

def melt_largo_para_longo(df_largo: pd.DataFrame, indicator_cols: List[str]) -> pd.DataFrame:
    # Garante meta cols presentes se existirem no Excel
    meta_available = [c for c in META_COLS if c in df_largo.columns]
    log.info(f"Meta cols: {meta_available}")
    log.info(f"Indicadores: {indicator_cols}")

    df = df_largo.copy()

    # Mês → número
    if "Mês" in df.columns:
        df["mes"] = df["Mês"].apply(_normalize_mes).astype("Int64")
    elif "Mes" in df.columns:
        df["mes"] = df["Mes"].apply(_normalize_mes).astype("Int64")
    else:
        df["mes"] = pd.NA

    # Cenário literal
    if "Cenário" in df.columns:
        df["cenario_label"] = df["Cenário"].astype(str).str.strip()
        df["cenario"] = df["cenario_label"]           # sem normalização
    else:
        df["cenario_label"] = "Realizado"
        df["cenario"] = "Realizado"

    # Ano: se existir coluna 'Ano' usamos; senão inferimos do cenario_label
    if "Ano" in df.columns:
        df["ano"] = pd.to_numeric(df["Ano"], errors="coerce").astype("Int64")
    else:
        df["ano"] = df["cenario_label"].apply(_infer_ano_from_cenario_literal).astype("Int64")

    # Renomeia campos utilitários (padroniza nomes sem acento para manter ambos)
    if "Cód. Produto" in df.columns:
        df["Cod_Produto"] = df["Cód. Produto"]
    if "Prod. Descrição" in df.columns:
        df["Prod_Descricao"] = df["Prod. Descrição"]

    # Melt para longo apenas dos indicadores
    id_vars = []
    for c in ["cenario", "cenario_label", "Sistema", "mes", "ano", "Cód. Produto", "Cod_Produto", "Prod. Descrição", "Prod_Descricao"]:
        if c in df.columns:
            id_vars.append(c)

    long_df = df[id_vars + indicator_cols].melt(
        id_vars=id_vars,
        value_vars=indicator_cols,
        var_name="indicador_nome",
        value_name="valor"
    )

    # Tipagem
    long_df["valor"] = pd.to_numeric(long_df["valor"], errors="coerce").fillna(0.0)
    long_df["ano"]   = pd.to_numeric(long_df["ano"], errors="coerce").astype("Int64")
    long_df["mes"]   = pd.to_numeric(long_df["mes"], errors="coerce").astype("Int64")

    # Mapeia indicador -> id canônico
    long_df["indicador_id"] = long_df["indicador_nome"].map(INDICATOR_MAP)
    long_df = long_df.dropna(subset=["indicador_id"]).copy()

    # Limpa linhas sem ano ou mês válidos
    before = len(long_df)
    long_df = long_df.dropna(subset=["ano", "mes"])
    after = len(long_df)
    if after < before:
        log.info(f"Descartadas {before - after} linha(s) sem ano/mês válido(s).")

    return long_df

def apply_familia_mapping(long_df: pd.DataFrame) -> pd.DataFrame:
    """Aplica de-para de Família Comercial se etl/familia/current.xlsx existir."""
    if not FAMILIA_FILE.exists():
        log.warning(f"[FAMÍLIA] Mapeamento não encontrado: {FAMILIA_FILE}")
        return long_df

    try:
        fam = pd.read_excel(FAMILIA_FILE)
    except Exception as e:
        log.warning(f"[FAMÍLIA] Erro ao ler '{FAMILIA_FILE}': {e}")
        return long_df

    # Procuramos chave por código do produto (ajuste se o seu arquivo usar outra coluna)
    key_cols = [c for c in fam.columns if "cód" in c.lower() or "cod" in c.lower()]
    fam_col  = [c for c in fam.columns if "família" in c.lower() or "familia" in c.lower()]

    if not key_cols or not fam_col:
        log.warning("[FAMÍLIA] Não encontrei colunas de chave/família no de-para; pulando mapeamento.")
        return long_df

    key = key_cols[0]
    fam_name = fam_col[0]
    fam = fam[[key, fam_name]].drop_duplicates()

    # Escolhe a coluna de chave presente no longo
    if "Cód. Produto" in long_df.columns:
        left_key = "Cód. Produto"
    elif "Cod_Produto" in long_df.columns:
        left_key = "Cod_Produto"
    else:
        log.warning("[FAMÍLIA] Coluna de código de produto não está presente no longo; pulando mapeamento.")
        return long_df

    merged = long_df.merge(fam, how="left", left_on=left_key, right_on=key)
    merged = merged.drop(columns=[key], errors="ignore")
    merged = merged.rename(columns={fam_name: "Família Comercial"})

    # Métrica de cobertura
    total = len(merged)
    com_fam = merged["Família Comercial"].notna().sum()
    if total > 0:
        pct = 100.0 * com_fam / total
        log.info(f"Família atribuída em {com_fam} / {total} ({pct:.1f}%).")

    return merged

def aggregate_grain(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega no grão final do Parquet:
    ano, mes, indicador_id, valor, cenario, cenario_label (+ campos auxiliares)
    """
    keys = ["ano", "mes", "indicador_id", "cenario", "cenario_label"]
    # Mantém colunas auxiliares se existirem (não entram no groupby)
    aux_cols = [c for c in ["Sistema", "Cód. Produto", "Cod_Produto", "Prod. Descrição", "Prod_Descricao", "Família Comercial"]
                if c in long_df.columns]

    g = long_df.groupby(keys, dropna=False, as_index=False)["valor"].sum()
    # Anexa algumas auxiliares como info (sem agregar) — opcional: aqui não repetimos porque a granularidade pode variar.
    return g

def save_parquet(df_final: pd.DataFrame) -> None:
    PARQUET_OUT.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    df_final.to_parquet(PARQUET_OUT, index=False)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    archive_path = ARCHIVE_DIR / f"{ts}_current.parquet"
    df_final.to_parquet(archive_path, index=False)

    log.info(f"[OK] Gravado:  {PARQUET_OUT}")
    log.info(f"[OK] Archive: {archive_path}")

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
def run():
    log.info(f"PROJECT_ROOT: {PROJECT_ROOT}")
    log.info(f"REALIZADO_RAW: {REALIZADO_RAW}")
    log.info(f"FAMILIA_FILE : {FAMILIA_FILE}")
    log.info(f"PARQUET_OUT  : {PARQUET_OUT}")

    etlres = load_realizado_largo()
    df_largo = etlres.df_long

    # Detecta indicadores disponíveis
    indicator_cols = detect_indicator_columns(df_largo)

    # LARGO -> LONGO
    long_df = melt_largo_para_longo(df_largo, indicator_cols)

    # (Opcional) Família Comercial
    long_df = apply_familia_mapping(long_df)

    # Agrega no grão final
    df_final = aggregate_grain(long_df)

    # Tipagem final
    df_final["ano"]   = pd.to_numeric(df_final["ano"], errors="coerce").astype("Int64")
    df_final["mes"]   = pd.to_numeric(df_final["mes"], errors="coerce").astype("Int64")
    df_final["valor"] = pd.to_numeric(df_final["valor"], errors="coerce").fillna(0.0)
    df_final["indicador_id"]  = df_final["indicador_id"].astype(str)
    df_final["cenario_label"] = df_final["cenario_label"].astype(str).str.strip()
    df_final["cenario"]       = df_final["cenario_label"]  # garantimos igualdade

    # Relatórios rápidos
    anos = sorted(df_final["ano"].dropna().unique().tolist())
    cenarios = df_final["cenario"].dropna().unique().tolist()
    log.info(f"Linhas finais: {len(df_final)}")
    log.info(f"Períodos (anos): {anos}")
    log.info(f"Cenários distintos (literais): {len(cenarios)} → amostra: {sorted(cenarios)[:10]}")

    # Salva
    save_parquet(df_final)

    log.info("ETL concluído com sucesso.")

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    run()
