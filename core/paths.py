# core/paths.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Raiz do projeto (SEM caminhos absolutos no código do app)
# -----------------------------------------------------------------------------
def _project_root() -> Path:
    """
    Descobre a raiz do projeto subindo pastas até achar 'data/'.
    Fallback: dois níveis acima do arquivo atual.
    """
    here = Path(__file__).resolve()

    # Tenta encontrar 'data/' subindo na hierarquia
    for p in [here] + list(here.parents):
        if (p / "data").exists():
            logger.info(f"Project root found at: {p}")
            return p

    # Fallback
    fallback = here.parents[1]
    logger.warning(f"Project root fallback to: {fallback}")
    return fallback


PROJECT_ROOT: Path = _project_root()

# -----------------------------------------------------------------------------
# Pastas principais
# -----------------------------------------------------------------------------
DATA_DIR: Path = PROJECT_ROOT / "data"

# Parquets
PARQUET_DIR: Path = DATA_DIR / "parquet"
CURRENT_PARQUET: Path = PARQUET_DIR / "current.parquet"
RES_WORKING_PARQUET: Path = PARQUET_DIR / "res_working.parquet"

# Premissas de P&L (preço/insumo/toll/frete etc.)
PREMISSAS_PNL_DIR: Path = DATA_DIR / "premissas_pnl"
BASE_CALCULOS_XLSX: Path = PREMISSAS_PNL_DIR / "base_calculos.xlsx"

# Capacidade / POH
CAP_POH_DIR: Path = DATA_DIR / "cap_poh"

# Estoque inicial
ESTOQUE_DIR: Path = DATA_DIR / "estoque"

# Produção (volumes planejados)
VOL_PROD_YTG_DIR: Path = DATA_DIR / "vol_prod_ytg"  # Produção futura (YTG)
VOL_PROD_YTD_DIR: Path = DATA_DIR / "vol_prod_ytd"  # Produção realizada (YTD)

# Modelos/Assets (para branding)
MODELS_DIR: Path = PROJECT_ROOT / "models"
LOGO_PATH: Path = MODELS_DIR / "logo.jpeg"


# -----------------------------------------------------------------------------
# Validação de estrutura
# -----------------------------------------------------------------------------
def validate_project_structure() -> Dict[str, bool]:
    """
    Valida se a estrutura de pastas e arquivos críticos existe.
    Retorna um dicionário com o status de cada item.
    """
    checks = {
        # Diretórios obrigatórios
        "data_dir": DATA_DIR.exists(),
        "parquet_dir": PARQUET_DIR.exists(),
        "premissas_pnl_dir": PREMISSAS_PNL_DIR.exists(),

        # Arquivos críticos
        "current_parquet": CURRENT_PARQUET.exists(),
        "base_calculos_xlsx": BASE_CALCULOS_XLSX.exists(),

        # Opcionais
        "res_working_parquet": RES_WORKING_PARQUET.exists(),
        "logo_file": LOGO_PATH.exists(),
        "models_dir": MODELS_DIR.exists(),
    }

    return checks


def create_missing_directories() -> List[str]:
    """
    Cria diretórios obrigatórios que não existem.
    Retorna lista de diretórios criados.
    """
    required_dirs = [
        DATA_DIR,
        PARQUET_DIR,
        PREMISSAS_PNL_DIR,
        CAP_POH_DIR,
        ESTOQUE_DIR,
        VOL_PROD_YTG_DIR,
        VOL_PROD_YTD_DIR,
        MODELS_DIR,
    ]

    created = []
    for dir_path in required_dirs:
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                created.append(str(dir_path))
                logger.info(f"Created directory: {dir_path}")
            except Exception as e:
                logger.error(f"Failed to create directory {dir_path}: {e}")

    return created


def get_project_status() -> Dict[str, any]:
    """
    Retorna status completo do projeto para diagnóstico.
    """
    validation = validate_project_structure()

    return {
        "project_root": str(PROJECT_ROOT),
        "validation": validation,
        "critical_missing": [
            k for k, v in validation.items()
            if not v and k in ["data_dir", "parquet_dir", "current_parquet"]
        ],
        "all_paths": path_summary(),
    }


def ensure_project_ready_for_streamlit() -> bool:
    """
    Valida estrutura do projeto para páginas Streamlit.
    Mostra erros na UI se houver problemas.
    """
    try:
        import streamlit as st
    except ImportError:
        # Se não está em contexto Streamlit, só retorna validação básica
        validation = validate_project_structure()
        critical = ["data_dir", "parquet_dir", "current_parquet"]
        return all(validation.get(k, False) for k in critical)

    validation = validate_project_structure()
    critical_issues = []

    # Verificar itens críticos
    if not validation.get("data_dir", False):
        critical_issues.append("Pasta 'data/' não encontrada")

    if not validation.get("parquet_dir", False):
        critical_issues.append("Pasta 'data/parquet/' não encontrada")

    if not validation.get("current_parquet", False):
        critical_issues.append("Arquivo 'current.parquet' não encontrado")

    if not validation.get("premissas_pnl_dir", False):
        critical_issues.append("Pasta 'data/premissas_pnl/' não encontrada")

    # Mostrar problemas se houver
    if critical_issues:
        st.error("🚨 **Projeto não está configurado corretamente!**")
        st.write("Problemas encontrados:")
        for issue in critical_issues:
            st.write(f"❌ {issue}")

        # Informações para debug
        with st.expander("🔍 Informações de Debug"):
            st.write("**Raiz do projeto detectada:**", PROJECT_ROOT)
            st.write("**Status da validação:**")
            for k, v in validation.items():
                icon = "✅" if v else "❌"
                st.write(f"{icon} {k}")

            st.write("**Todos os caminhos:**")
            for k, v in path_summary().items():
                st.code(f"{k}: {v}")

        return False

    return True


# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------
def path_summary() -> Dict[str, str]:
    """Resumo de caminhos para logs/depuração."""
    return {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "DATA_DIR": str(DATA_DIR),
        "PARQUET_DIR": str(PARQUET_DIR),
        "CURRENT_PARQUET": str(CURRENT_PARQUET),
        "RES_WORKING_PARQUET": str(RES_WORKING_PARQUET),
        "PREMISSAS_PNL_DIR": str(PREMISSAS_PNL_DIR),
        "BASE_CALCULOS_XLSX": str(BASE_CALCULOS_XLSX),
        "CAP_POH_DIR": str(CAP_POH_DIR),
        "ESTOQUE_DIR": str(ESTOQUE_DIR),
        "VOL_PROD_YTG_DIR": str(VOL_PROD_YTG_DIR),
        "VOL_PROD_YTD_DIR": str(VOL_PROD_YTD_DIR),
        "MODELS_DIR": str(MODELS_DIR),
        "LOGO_PATH": str(LOGO_PATH),
    }


def safe_file_check(file_path: Path) -> Tuple[bool, str]:
    """
    Verifica se arquivo existe e retorna informações úteis.
    Retorna (exists, info_message).
    """
    try:
        if file_path.exists():
            stat = file_path.stat()
            size_mb = stat.st_size / (1024 * 1024)
            return True, f"Found ({size_mb:.1f} MB)"
        else:
            return False, f"Missing: {file_path}"
    except Exception as e:
        return False, f"Error checking {file_path}: {e}"


__all__ = [
    # Paths principais
    "PROJECT_ROOT",
    "DATA_DIR",
    "PARQUET_DIR",
    "CURRENT_PARQUET",
    "RES_WORKING_PARQUET",
    "PREMISSAS_PNL_DIR",
    "BASE_CALCULOS_XLSX",
    "CAP_POH_DIR",
    "ESTOQUE_DIR",
    "VOL_PROD_YTG_DIR",
    "VOL_PROD_YTD_DIR",
    "MODELS_DIR",
    "LOGO_PATH",

    # Utilities
    "path_summary",
    "validate_project_structure",
    "create_missing_directories",
    "get_project_status",
    "safe_file_check",
    "ensure_project_ready_for_streamlit",
]