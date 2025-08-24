# etl/main.py
# -*- coding: utf-8 -*-
"""
CLI da ETL ÚNICA (governança + baixo TCO)

Blocos:
- etl_realizado  -> current.parquet      (oficial/realizado + cenários)
- etl_projecao   -> res_working.parquet  (projeções/volumes de trabalho)

Uso:
  python -m simulador_dre_streamlit.etl.main --only all
  python -m simulador_dre_streamlit.etl.main --only realizado
  python -m simulador_dre_streamlit.etl.main --only projecao
  python -m simulador_dre_streamlit.etl.main --log-level DEBUG
  python -m simulador_dre_streamlit.etl.main --no-manifest
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from .common import (
    setup_logger,
    DATA_DIR,
    PARQUET_DIR,
    ARCHIVE_DIR,
    CURRENT_PARQUET,
    RES_WORKING_PARQUET,
)
from . import pipelines


def _print_final_shapes(log: logging.Logger, result: Dict[str, Any]) -> None:
    """Imprime shapes dos artefatos gerados (sanity check)."""
    try:
        cur = result.get("current", {})
        df = cur.get("df")
        if df is None and CURRENT_PARQUET.exists():
            df = pd.read_parquet(CURRENT_PARQUET)
        if df is not None:
            log.info(f"[CURRENT] {CURRENT_PARQUET} -> shape={df.shape}")
        else:
            log.warning(f"[CURRENT] Arquivo não encontrado: {CURRENT_PARQUET}")
    except Exception as e:
        log.warning(f"[CURRENT] Falha ao ler {CURRENT_PARQUET}: {e}")

    try:
        res = result.get("res_working", {})
        df = res.get("df")
        if df is None and RES_WORKING_PARQUET.exists():
            df = pd.read_parquet(RES_WORKING_PARQUET)
        if df is not None:
            log.info(f"[RES] {RES_WORKING_PARQUET} -> shape={df.shape}")
        else:
            log.warning(f"[RES] Arquivo não encontrado: {RES_WORKING_PARQUET}")
    except Exception as e:
        log.warning(f"[RES] Falha ao ler {RES_WORKING_PARQUET}: {e}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ETL Unificada (realizado + projeção)")
    parser.add_argument(
        "--only",
        choices=("all", "realizado", "projecao"),
        default="all",
        help="Qual bloco executar (default: all).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Nível de log (default: INFO).",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Não gerar arquivos .manifest.json ao lado dos parquets.",
    )
    args = parser.parse_args(argv)

    level = getattr(logging, args.log_level, logging.INFO)
    log = setup_logger("etl.main", level)

    log.info("============================================")
    log.info("          ETL UNIFICADA — INÍCIO            ")
    log.info("============================================")
    log.info(f"Base de dados (Excel): {DATA_DIR}")
    log.info(f"Saída (parquets):      {PARQUET_DIR}")
    log.info(f"Archive:               {ARCHIVE_DIR}")

    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {"current": {"df": None}, "res_working": {"df": None}}
    try:
        if args.only == "realizado":
            path, df = pipelines.etl_realizado(write_manifest_flag=not args.no_manifest, logger=log)
            result["current"] = {"path": path, "df": df}
        elif args.only == "projecao":
            path, df = pipelines.etl_projecao(write_manifest_flag=not args.no_manifest, logger=log)
            result["res_working"] = {"path": path, "df": df}
        else:
            result = pipelines.run_all(logger=log)
            # Se pediram --no-manifest, reexecutar summaries não é necessário; só avisar
            if args.no_manifest:
                log.info("[GOVERNANÇA] Execução completa sem manifest (--no-manifest).")
    except Exception as e:
        log.exception("Falha na execução da ETL unificada.")
        return 1

    _print_final_shapes(log, result)

    log.info("============================================")
    log.info("         ETL UNIFICADA — CONCLUÍDA          ")
    log.info("============================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
