# etl/pipelines.py
# -*- coding: utf-8 -*-
"""
PIPELINES UNIFICADAS (sem dependência dos legados):
- etl_realizado  -> current.parquet      (oficial/realizado + cenários)
- etl_projecao   -> res_working.parquet  (projeções/volumes de trabalho)

Arquitetura:
- Leitura de Excel: SEMPRE sob `data/` (governança).
- Transformações de negócio: `etl/transformers.py`.
- Classificação ABSOLUTA por SKU: `etl/common.classify_familia` (mestre substitui qualquer valor).
- Escrita: atômica + archive, manifest opcional, e resumos executivos no console.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

# Núcleo compartilhado (single source of truth)
from .common import (
    setup_logger,
    DATA_DIR,
    PARQUET_DIR,
    ARCHIVE_DIR,
    CURRENT_PARQUET,
    RES_WORKING_PARQUET,
    MASTER_SKU_PATH,
    read_excel_safe,          # leitura com alerta caso fora de `data/`
    classify_familia,         # mestre ABSOLUTO por SKU + fallback de-para
    atomic_write_parquet,     # escrita atômica + archive
    write_manifest,           # manifest JSON opcional
    print_current_summary,    # resumo por cenário
    print_res_working_summary # resumo RE (anos, famílias, volume)
)

# Regras de negócio (puras, sem I/O)
from . import transformers as T


# =============================================================================
# Defaults (padrões de caminho sob data/)
# =============================================================================

# Realizado: ler todos os Excel(s) de data/realizado/
REALIZADO_DIR_DEFAULT = DATA_DIR / "realizado"
# Projeção: arquivo único padrão
PROJECAO_XLSX_DEFAULT = DATA_DIR / "res" / "working_re_volumes.xlsx"
# De-para (opcional) — manter em data/familia (sem acento, conforme projeto atual)
DEPARA_FAMILIA_DEFAULT = DATA_DIR / "familia" / "re_depara_familia.xlsx"


# =============================================================================
# Helpers (I/O de entrada)
# =============================================================================

def _list_excel_files(folder: Path, patterns: Sequence[str] = ("*.xlsx", "*.xls")) -> List[Path]:
    """
    Lista arquivos Excel em `folder` respeitando os padrões informados.
    Padrão inclui .xlsx e .xls (xlrd/openpyxl conforme o tipo).
    """
    out: List[Path] = []
    for pat in patterns:
        out.extend(sorted(folder.glob(pat)))
    # resolve e dedup
    return sorted({p.resolve() for p in out})

def _read_all_excels(paths: Sequence[Path], sheet_name: Optional[str], logger) -> pd.DataFrame:
    """
    Lê e concatena uma lista de Excel(s), preservando colunas.
    - Usa `read_excel_safe` (governança: alerta se for fora de `data/`).
    - Se `sheet_name` não for informado, usa SEMPRE a PRIMEIRA aba (sheet=0).
    - Adiciona coluna `_source_file` para auditoria/manifest.
    - Concatena com `outer` para tolerar variações de schema entre arquivos.
    """
    frames: List[pd.DataFrame] = []
    for p in paths:
        effective_sheet = sheet_name if sheet_name is not None else 0
        obj = read_excel_safe(p, sheet_name=effective_sheet, dtype="object", logger=logger)
        # Robustez: se algum engine devolver dict por qualquer motivo, usa a primeira aba
        if isinstance(obj, dict):
            first_name, df = next(iter(obj.items()))
            logger.info(f"[READ] {p} -> usando primeira aba detectada: {first_name}")
        else:
            df = obj
        df["_source_file"] = str(p)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)


# =============================================================================
# BLOCO 1 — ETL REALIZADO (current.parquet)
# =============================================================================

def etl_realizado(
    *,
    realizado_dir: Path = REALIZADO_DIR_DEFAULT,
    realizado_patterns: Sequence[str] = ("*.xlsx", "*.xls"),
    realizado_sheet: Optional[str] = None,
    depara_path: Optional[Path] = DEPARA_FAMILIA_DEFAULT,
    master_sku_path: Path = MASTER_SKU_PATH,
    write_manifest_flag: bool = True,
    logger=None,
) -> Tuple[Path, Optional[pd.DataFrame]]:
    """
    Executa a ETL do REALIZADO/CENÁRIOS (current.parquet):

      1) Ler Excel(s) em `data/realizado/` (varre *.xlsx e *.xls).
      2) Transformar via `transformers.transform_realizado(...)`.
      3) Classificação/Normalização:
      - Se houver SKU (não é o caso padrão do RES), usa mestre absoluto;
            - Sem SKU, normaliza por 'Família Comercial' (canoniza contra o mestre quando possível).
         (mestre substitui qualquer 'Família Comercial'; SKU normalizado para TEXTO).
      4) Escrever com `atomic_write_parquet` + `archive/`, opcionalmente gerar `manifest`.
      5) Imprimir **resumo executivo** por cenário no console.

    Retorna: (caminho_do_parquet, dataframe_final | None)
    """
    log = logger or setup_logger("etl.pipelines")

    # 1) Descobrir & ler fontes
    src_files = _list_excel_files(realizado_dir, realizado_patterns)
    if not src_files:
        log.error(f"[CURRENT] Nenhum Excel encontrado em: {realizado_dir} (padrões: {list(realizado_patterns)})")
        return CURRENT_PARQUET, None

    log.info(f"[CURRENT] Lendo {len(src_files)} arquivo(s) de {realizado_dir}…")
    for p in src_files:
        log.info(f"    - {p}")
    raw_df = _read_all_excels(src_files, sheet_name=realizado_sheet, logger=log)
    if raw_df.empty:
        log.error("[CURRENT] Leitura retornou vazio.")
        return CURRENT_PARQUET, None

    # 2) Transformações de negócio
    log.info("[CURRENT] Aplicando transformações (DRE completo → métricas do P&L)…")
    tr_df = T.transform_realizado(raw_df)

    # --- Sanity / Telemetria pós-transform ---
    # (ajuste) incluir 'Tecnologia' em dims quando existir — apenas para log/telemetria
    dims = [c for c in ["SKU", "Família Comercial", "Tecnologia", "produto_descricao",
                        "sistema", "cenario", "ano", "mes"] if c in tr_df.columns]
    metrics = [c for c in tr_df.columns
               if c not in dims and pd.api.types.is_numeric_dtype(tr_df[c])]

    log.info(f"[CURRENT] Transform OK → shape={tr_df.shape} | dims={len(dims)} | metrics={len(metrics)}")

    if "cenario" in tr_df.columns:
        scen = tr_df["cenario"].dropna().astype(str).unique().tolist()
        preview = ", ".join(sorted(scen)[:5])
        log.info(f"[CURRENT] Cenários detectados ({len(scen)}): {preview}{' ...' if len(scen) > 5 else ''}")

    if "ano" in tr_df.columns:
        anos = sorted([int(x) for x in tr_df["ano"].dropna().unique()])
        log.info(f"[CURRENT] Períodos (anos): {anos}")

    if tr_df.empty:
        log.warning("[CURRENT] Transform resultou em DataFrame vazio — revise aliases/headers do arquivo.")

    # 3) Classificação ABSOLUTA por SKU (mestre > tudo; de-para = fallback)
    final_df = classify_familia(
        tr_df,
        sku_col_hint="SKU",
        depara_path=depara_path,
        master_sku_path=master_sku_path,
        logger=log,
        log_stats=True,
    )

    # 4) Escrita + archive
    out_path, archive_path = atomic_write_parquet(final_df, CURRENT_PARQUET, ARCHIVE_DIR)
    log.info(f"[CURRENT] Gravado: {out_path}")
    log.info(f"[CURRENT] Archive: {archive_path}")

    # Manifest (opcional)
    if write_manifest_flag:
        extra = {
            "dataset": "current",
            "pipeline": "etl_realizado",
            "sources": [str(p) for p in src_files],
            "sheet": realizado_sheet or "(first/active)",
        }
        write_manifest(final_df, out_path, extra=extra, logger=log)

    # 5) Resumo executivo
    print_current_summary(final_df, logger=log)

    return out_path, final_df


# =============================================================================
# BLOCO 2 — ETL PROJEÇÃO (res_working.parquet)
# =============================================================================

def etl_projecao(
    *,
    projecao_xlsx: Path = PROJECAO_XLSX_DEFAULT,
    projecao_sheet: Optional[str] = None,
    depara_path: Optional[Path] = DEPARA_FAMILIA_DEFAULT,
    master_sku_path: Path = MASTER_SKU_PATH,
    write_manifest_flag: bool = True,
    logger=None,
) -> Tuple[Path, Optional[pd.DataFrame]]:
    """
    Executa a ETL das PROJEÇÕES/RE working (res_working.parquet):

      1) Ler `data/res/working_re_volumes.xlsx` (ou conforme parâmetros).
      2) Transformar via `transformers.transform_projecao(...)` (melt meses → long).
      3) Classificar **ABSOLUTAMENTE** por SKU via `common.classify_familia(...)`
         (mestre substitui qualquer 'Família Comercial'; SKU normalizado para TEXTO).
      4) Escrever com `atomic_write_parquet` + `archive/`, opcionalmente gerar `manifest`.
      5) Imprimir **resumo executivo** no console.

    Retorna: (caminho_do_parquet, dataframe_final | None)
    """
    log = logger or setup_logger("etl.pipelines")

    # 1) Ler fonte única (working). Se a aba não for informada, usar a PRIMEIRA (sheet=0)
    if not Path(projecao_xlsx).exists():
        log.error(f"[RES] Arquivo não encontrado: {projecao_xlsx}")
        return RES_WORKING_PARQUET, None

    log.info(f"[RES] Lendo {projecao_xlsx}…")
    effective_sheet = projecao_sheet if projecao_sheet is not None else 0
    obj = read_excel_safe(projecao_xlsx, sheet_name=effective_sheet, dtype="object", logger=log)
    if isinstance(obj, dict):
        first_name, wide_df = next(iter(obj.items()))
        log.info(f"[READ] {projecao_xlsx} -> usando primeira aba detectada: {first_name}")
    else:
        wide_df = obj

    if wide_df.empty:
        log.error("[RES] Leitura retornou vazio.")
        return RES_WORKING_PARQUET, None

    # 2) Transformações de negócio (melt → long)
    long_df = T.transform_projecao(wide_df)

    # 3) Classificação ABSOLUTA por SKU (mestre > tudo; de-para = fallback)
    final_df = classify_familia(
        long_df,
        sku_col_hint="SKU",
        depara_path=depara_path,
        master_sku_path=master_sku_path,
        logger=log,
        log_stats=True,
    )

    # 4) Escrita + archive
    out_path, archive_path = atomic_write_parquet(final_df, RES_WORKING_PARQUET, ARCHIVE_DIR)
    log.info(f"[RES] Gravado: {out_path}")
    log.info(f"[RES] Archive: {archive_path}")

    # Manifest (opcional)
    if write_manifest_flag:
        extra = {
            "dataset": "res_working",
            "pipeline": "etl_projecao",
            "sources": [str(projecao_xlsx)],
            "sheet": projecao_sheet or "(first/active)",
        }
        write_manifest(final_df, out_path, extra=extra, logger=log)

    # 5) Resumo executivo
    print_res_working_summary(final_df, logger=log)

    return out_path, final_df


# =============================================================================
# ORQUESTRADOR — EXECUTA OS DOIS BLOCOS
# =============================================================================

def run_all(logger=None) -> Dict[str, Any]:
    """
    Executa `etl_realizado` e `etl_projecao` na sequência.
    Retorna metadados úteis para smoke tests e integrações downstream.
    """
    log = logger or setup_logger("etl.pipelines")
    log.info("============================================")
    log.info("          ETL UNIFICADA — INÍCIO            ")
    log.info("============================================")

    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    cur_path, cur_df = etl_realizado(logger=log)
    res_path, res_df = etl_projecao(logger=log)

    log.info("============================================")
    log.info("         ETL UNIFICADA — CONCLUÍDA          ")
    log.info("============================================")

    return {
        "current": {"path": cur_path, "df": cur_df},
        "res_working": {"path": res_path, "df": res_df},
    }
