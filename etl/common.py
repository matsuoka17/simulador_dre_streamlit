# etl/common.py
# -*- coding: utf-8 -*-
"""
NÚCLEO COMPARTILHADO DA ETL (governança e baixo TCO)

Este módulo concentra utilidades e regras transversais:
- Paths padronizados (data/parquet/)
- Logger único
- Leitura robusta de Excel (somente em data/)
- Normalizações (mês, ano, tipagem, SKU como TEXTO)
- De/Para de Família (fallback)
- Classificação ABSOLUTA por SKU via mestre (quando SKU existe)
- Normalização por Família Comercial quando NÃO existe SKU
- Escrita atômica em parquet + archive com timestamp
- Manifest JSON de governança
- Resumos executivos (current e res_working)
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Dict

import numpy as np
import pandas as pd


# =============================================================================
# PATHS PADRÃO (alinhado: data/parquet/)
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"                     # Governança: entradas devem estar em data/
PARQUET_DIR  = DATA_DIR / "parquet"
ARCHIVE_DIR  = PARQUET_DIR / "archive"

CURRENT_PARQUET     = PARQUET_DIR / "current.parquet"
RES_WORKING_PARQUET = PARQUET_DIR / "res_working.parquet"

# Mestre absoluto de SKU → Família Comercial (projeto atual usa 'familia' sem acento)
MASTER_SKU_PATH = DATA_DIR / "familia" / "current.xlsx"


# =============================================================================
# LOGGER ÚNICO
# =============================================================================

def setup_logger(name: str = "etl", level: int = logging.INFO) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        log.addHandler(handler)
    log.setLevel(level)
    return log


# =============================================================================
# HELPERS DE NORMALIZAÇÃO / MATCH
# =============================================================================

def _strip_accents_lower(s: str) -> str:
    m = (
        ("á", "a"), ("à", "a"), ("â", "a"), ("ã", "a"),
        ("é", "e"), ("ê", "e"),
        ("í", "i"),
        ("ó", "o"), ("ô", "o"), ("õ", "o"),
        ("ú", "u"),
        ("ç", "c"),
    )
    s2 = str(s).lower()
    for a, b in m:
        s2 = s2.replace(a, b)
    return s2

def _norm_colname(c: str) -> str:
    return _strip_accents_lower(c).strip()

def find_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    canon = { _norm_colname(c): c for c in df.columns }
    for cand in candidates:
        key = _norm_colname(cand)
        if key in canon:
            return canon[key]
        # heurística leve para "sku"
        for k, real in canon.items():
            if key == "sku" and ("sku" in k.split() or k.endswith("sku") or k.startswith("sku")):
                return real
    return None


# =============================================================================
# NORMALIZAÇÕES E VALIDAÇÕES
# =============================================================================

_MONTHS_MAP = {
    "jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6,
    "jul": 7, "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12,
    "janeiro": 1, "fevereiro": 2, "marco": 3, "março": 3, "abril": 4, "maio": 5, "junho": 6,
    "julho": 7, "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
    "jan.": 1, "feb": 2, "feb.": 2, "mar.": 3, "apr": 4, "apr.": 4, "may": 5, "jun.": 6,
    "jul.": 7, "aug": 8, "aug.": 8, "sep": 9, "sep.": 9, "oct": 10, "oct.": 10, "nov.": 11, "dec": 12, "dec.": 12,
}

def normalize_mes(val) -> Optional[int]:
    if pd.isna(val):
        return None
    try:
        m = int(str(val).strip())
        if 1 <= m <= 12:
            return m
    except Exception:
        pass
    s = _strip_accents_lower(str(val).strip()).replace(".", "")
    m = re.search(r"(\d{1,2})", s)
    if s.startswith("m") and m:
        mm = int(m.group(1))
        return mm if 1 <= mm <= 12 else None
    if s in _MONTHS_MAP:
        return _MONTHS_MAP[s]
    tokens = re.split(r"[\s_\-\/]+", s)
    for t in reversed(tokens):
        if t in _MONTHS_MAP:
            return _MONTHS_MAP[t]
        if t.isdigit():
            v = int(t)
            if 1 <= v <= 12:
                return v
    return None

def to_int_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")

def ensure_columns(df: pd.DataFrame, required: Sequence[str], ctx: str = "") -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[Schema] Colunas faltantes {missing} em {ctx or 'dataframe'}.")

def extract_years(df: pd.DataFrame) -> List[int]:
    if "ano" not in df.columns:
        return []
    return sorted(set(pd.to_numeric(df["ano"], errors="coerce").dropna().astype(int).tolist()))


# =============================================================================
# SKU → TEXTO (NORMALIZAÇÃO)
# =============================================================================

def normalize_sku_text(series: pd.Series) -> pd.Series:
    def _norm(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip()
        if re.fullmatch(r"\d+\.0+", s) or re.fullmatch(r"\d+\.(0)+", s):
            return s.split(".")[0]
        return s
    return series.map(_norm)


# =============================================================================
# LEITURA DE EXCEL + MELT SEGURO (GOVERNANÇA: SOMENTE EM data/)
# =============================================================================

def _assert_path_under_data(path: Path, log: Optional[logging.Logger] = None) -> None:
    log = log or setup_logger("etl.common")
    try:
        p = Path(path).resolve()
        if str(DATA_DIR.resolve()) not in str(p):
            log.warning(f"[GOVERNANÇA] Excel fora de `data/`: {p}. "
                        f"Padronize leituras sob `{DATA_DIR}`.")
    except Exception:
        log.warning(f"[GOVERNANÇA] Não foi possível validar path: {path}")

def read_excel_safe(path: Path, sheet_name: Optional[str] = None, dtype="object", logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    log = logger or setup_logger("etl.common")
    _assert_path_under_data(path, log)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo Excel não encontrado: {path}")
    return pd.read_excel(path, sheet_name=sheet_name, dtype=dtype)

def safe_melt_months(
    df_wide: pd.DataFrame,
    id_vars: Sequence[str],
    month_candidates: Optional[Iterable[str]] = None,
    value_name: str = "valor",
) -> pd.DataFrame:
    id_vars = list(id_vars)
    cols = [c for c in df_wide.columns if c not in id_vars] if month_candidates is None else list(month_candidates)
    long_df = df_wide.melt(id_vars=id_vars, value_vars=cols, var_name="mes_raw", value_name=value_name)
    long_df["mes"] = long_df["mes_raw"].map(normalize_mes)
    long_df = long_df.dropna(subset=["mes"]).copy()
    long_df["mes"] = long_df["mes"].astype(int)
    return long_df.drop(columns=["mes_raw"])


# =============================================================================
# DE/PARA DE FAMÍLIA (LEGADO, USADO COMO FALLBACK)
# =============================================================================

def apply_depara_familia(
    df_long: pd.DataFrame,
    depara_path: Optional[Path],
    source_col: str = "Família Gerencial",
    target_col: str = "Família Comercial",
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    log = logger or setup_logger("etl.common")
    if target_col in df_long.columns:
        return df_long
    if depara_path and Path(depara_path).exists():
        try:
            de = pd.read_excel(depara_path)
            cols = [str(c).strip() for c in de.columns]
            de.columns = cols

            def col_match(token: str) -> Optional[str]:
                token = _strip_accents_lower(token)
                for c in cols:
                    cl = _strip_accents_lower(c)
                    if token in cl:
                        return c
                return None

            cand_ger = col_match("familia ger")
            cand_com = col_match("familia com")
            if cand_ger and cand_com:
                de = de[[cand_ger, cand_com]].dropna().drop_duplicates()
                de = de.rename(columns={cand_ger: source_col, cand_com: target_col})
                out = df_long.merge(de, how="left", on=source_col)
                out[target_col] = out[target_col].fillna(out[source_col])
                out = out.drop(columns=[source_col])
                return out
            else:
                log.warning("[de-para] Colunas de mapeamento não encontradas; usando origem como Comercial.")
        except Exception as e:
            log.warning(f"[de-para] Falha ao aplicar de-para: {e}; usando origem como Comercial.")
    if source_col in df_long.columns:
        return df_long.rename(columns={source_col: target_col})
    return df_long


# =============================================================================
# MESTRE ABSOLUTO (SKU → FAMÍLIA COMERCIAL)
# =============================================================================

def load_master_sku(master_path: Path = MASTER_SKU_PATH, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    log = logger or setup_logger("etl.common")
    _assert_path_under_data(master_path, log)

    if not Path(master_path).exists():
        raise FileNotFoundError(f"[MESTRE SKU] Arquivo não encontrado: {master_path}")

    df = pd.read_excel(master_path, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]

    sku_col = find_column(df, [
        "SKU","sku","Sku",
        "Código SKU","Codigo SKU","Cod SKU","CodSku",
        "Cód. Produto","Cod. Produto","Código do Produto","Codigo do Produto",
        "Código Produto","Codigo Produto","Cod Produto"
    ])
    fam_com_col = find_column(df, ["Família Comercial", "Familia Comercial"])
    fam_ger_col = find_column(df, ["Família Gerencial", "Familia Gerencial"])

    req = []
    if not sku_col:
        req.append("SKU (ou equivalente)")
    if not fam_com_col and not fam_ger_col:
        req.append("Família Comercial ou Família Gerencial")
    if req:
        raise ValueError(f"[MESTRE SKU] Colunas obrigatórias ausentes: {req} em {master_path}")

    df[sku_col] = normalize_sku_text(df[sku_col])

    out = df[[sku_col]].copy()
    if fam_com_col:
        out["Família Comercial"] = df[fam_com_col].astype(str).str.strip()
    elif fam_ger_col:
        out["Família Gerencial"] = df[fam_ger_col].astype(str).str.strip()

    dup = out.duplicated(subset=[sku_col], keep=False).sum()
    if dup > 0:
        log.warning(f"[MESTRE SKU] SKUs duplicados no mestre: {dup} linha(s). "
                    f"Considere deduplicar {master_path}.")
        out = out.drop_duplicates(subset=[sku_col], keep="first")

    out = out.rename(columns={sku_col: "SKU"})
    return out


def classify_familia(
    df: pd.DataFrame,
    *,
    sku_col_hint: Optional[str] = "SKU",
    depara_path: Optional[Path] = None,
    master_sku_path: Path = MASTER_SKU_PATH,
    logger: Optional[logging.Logger] = None,
    log_stats: bool = True,
) -> pd.DataFrame:
    """
    Classifica/normaliza 'Família Comercial'.
    CASO A) Com SKU: mestre por SKU **SEMPRE prevalece** (substitui qualquer valor existente).
    CASO B) Sem SKU: normaliza 'Família Comercial' (trim/case) e canoniza contra o mestre (se disponível).
                Antes disso, tenta de-para a partir de 'Família Gerencial' (se existir).
    """
    log = logger or setup_logger("etl.common")
    df = df.copy()

    # 1) Detecta coluna de SKU (aliases ampliados, incluindo 'Cód. Produto')
    sku_col = sku_col_hint or "SKU"
    if sku_col not in df.columns:
        found = find_column(df, [
            "SKU","sku","Sku",
            "Código SKU","Codigo SKU","Cod SKU","CodSku",
            "Cód. Produto","Cod. Produto",
            "Código do Produto","Codigo do Produto",
            "Código Produto","Codigo Produto","Cod Produto"
        ])
        if not found:
            # ===== CASO B: SEM SKU =====
            log.warning("[CLASSIF] Coluna de SKU não encontrada na fonte. "
                        "Normalizando por 'Família Comercial' (com de-para e canonização quando possível).")
            # 1) Tenta garantir 'Família Comercial' via de-para (se vier 'Família Gerencial')
            df = apply_depara_familia(df, depara_path, logger=log)
            fam_com_col = find_column(df, ["Família Comercial", "Familia Comercial"])

            if fam_com_col:
                if fam_com_col != "Família Comercial":
                    df = df.rename(columns={fam_com_col: "Família Comercial"})
                df["Família Comercial"] = df["Família Comercial"].astype(str).str.strip()

                # 2) Canoniza contra as famílias do mestre (match exato normalizado)
                try:
                    master = load_master_sku(master_sku_path, logger=log)
                    if "Família Comercial" in master.columns:
                        canon_map = {}
                        for v in master["Família Comercial"].dropna().astype(str).unique():
                            key = _strip_accents_lower(v).strip()
                            canon_map.setdefault(key, v)
                        df["_fam_key"] = df["Família Comercial"].map(lambda s: _strip_accents_lower(str(s)).strip())
                        df["Família Comercial"] = df["_fam_key"].map(canon_map).fillna(df["Família Comercial"])
                        df = df.drop(columns=["_fam_key"])
                except Exception:
                    pass
            else:
                log.warning("[CLASSIF] Nem 'Família Comercial' nem 'Família Gerencial' disponíveis; "
                            "não foi possível classificar.")

            if log_stats:
                total = len(df)
                fam_na = df["Família Comercial"].isna().sum() if "Família Comercial" in df.columns else total
                log.info(f"[CLASSIF] (SEM SKU) Linhas: {total} | Sem família após normalização: {fam_na}")
            return df
        sku_col = found

    # ===== CASO A: COM SKU =====
    df[sku_col] = normalize_sku_text(df[sku_col])
    master = load_master_sku(master_sku_path, logger=log)

    has_com = "Família Comercial" in master.columns
    has_ger = "Família Gerencial" in master.columns

    joined = df.merge(master, how="left", left_on=sku_col, right_on="SKU")

    if has_com:
        # Força substituição absoluta
        if "Família Comercial" in joined.columns and "Família Comercial_y" in joined.columns:
            joined = joined.drop(columns=["Família Comercial"])
        if "Família Comercial_y" in joined.columns:
            joined = joined.rename(columns={"Família Comercial_y": "Família Comercial"})
        # remove sobras
        for col in ("Família Comercial_x", "SKU_y"):
            if col in joined.columns:
                joined = joined.drop(columns=[col])
        if "SKU_x" in joined.columns:
            joined = joined.rename(columns={"SKU_x": sku_col})
    else:
        # Mestre só possui Gerencial → manter/combinar para de-para
        if "Família Gerencial_y" in joined.columns:
            if "Família Gerencial" not in joined.columns:
                joined["Família Gerencial"] = joined["Família Gerencial_y"]
            else:
                joined["Família Gerencial"] = joined["Família Gerencial"].fillna(joined["Família Gerencial_y"])
        for col in ("Família Gerencial_x", "Família Gerencial_y", "SKU_y"):
            if col in joined.columns:
                joined = joined.drop(columns=[col])
        if "SKU_x" in joined.columns:
            joined = joined.rename(columns={"SKU_x": sku_col})

    # Fallback final: se ainda faltar 'Família Comercial', tenta de-para
    if "Família Comercial" not in joined.columns or joined["Família Comercial"].isna().any():
        joined = apply_depara_familia(joined, depara_path, logger=log)
        fam_com_col = find_column(joined, ["Família Comercial", "Familia Comercial"])
        if fam_com_col and fam_com_col != "Família Comercial":
            joined = joined.rename(columns={fam_com_col: "Família Comercial"})

    if log_stats:
        total = len(joined)
        via_master = joined["Família Comercial"].notna().sum() if "Família Comercial" in joined.columns else 0
        sku_present = joined[sku_col].notna().sum()
        not_classified = total - via_master
        log.info(f"[CLASSIF] (COM SKU) Linhas: {total} | SKU presente: {sku_present} | "
                 f"Classificadas via MESTRE: {via_master} | Sem família pos-fallback: {not_classified}")
        if not_classified > 0:
            sample = joined[joined["Família Comercial"].isna()].head(5)
            cols_show = [sku_col]
            ger_col = find_column(joined, ["Família Gerencial", "Familia Gerencial"])
            if ger_col:
                cols_show.append(ger_col)
            log.info("[CLASSIF] Amostra de não classificados (top 5):")
            try:
                log.info(sample[cols_show].to_string(index=False))
            except Exception:
                log.info(sample.head(5).to_string(index=False))

    return joined


# =============================================================================
# ESCRITA ATÔMICA + ARCHIVE + MANIFEST
# =============================================================================

def atomic_write_parquet(df: pd.DataFrame, out_path: Path, archive_dir: Path) -> Tuple[Path, Path]:
    out_path = Path(out_path)
    archive_dir = Path(archive_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, dir=str(out_path.parent), suffix=".parquet") as tmp:
        tmp_name = tmp.name
    try:
        df.to_parquet(tmp_name, index=False)
        os.replace(tmp_name, out_path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except Exception:
                pass

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    archive_path = archive_dir / f"{ts}_{out_path.name}"
    df.to_parquet(archive_path, index=False)
    return out_path, archive_path

def write_manifest(
    df: pd.DataFrame,
    out_path: Path,
    extra: Optional[Dict] = None,
    logger: Optional[logging.Logger] = None,
) -> Path:
    log = logger or setup_logger("etl.common")
    manifest = {
        "file": str(out_path),
        "row_count": int(len(df)),
        "schema": {c: str(dt) for c, dt in df.dtypes.items()},
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if "ano" in df.columns:
        anos = extract_years(df)
        manifest["anos"] = anos
    if "mes" in df.columns:
        meses = pd.to_numeric(df["mes"], errors="coerce").dropna().astype(int)
        if not meses.empty:
            manifest["mes_min"] = int(meses.min())
            manifest["mes_max"] = int(meses.max())
    fam_com_col = find_column(df, ["Família Comercial", "Familia Comercial"])
    if fam_com_col:
        fams = sorted(set(df[fam_com_col].dropna().astype(str)))
        manifest["familias_distintas"] = len(fams)
    scen_col = scenario_column(df)
    if scen_col:
        scens = sorted(set(df[scen_col].dropna().astype(str)))
        manifest["cenarios_distintos"] = len(scens)
    if extra:
        manifest.update(extra)
    path = Path(str(out_path) + ".manifest.json")
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"[MANIFEST] {path}")
    return path


# =============================================================================
# RESUMOS EXECUTIVOS (CONSOLE)
# =============================================================================

def _fmt_brl(n: float) -> str:
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "-"
    s = f"{float(n):,.2f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def scenario_column(df: pd.DataFrame) -> Optional[str]:
    if "cenario" in df.columns:
        return "cenario"
    if "Cenário" in df.columns:
        return "Cenário"
    return None

def summarize_current_by_cenario(df_current: pd.DataFrame) -> pd.DataFrame:
    if df_current is None or df_current.empty:
        return pd.DataFrame(columns=["cenario", "volume_uc", "receita_liquida", "resultado_operacional"])
    scen_col = scenario_column(df_current) or "cenario"
    grp = df_current.copy()
    vol_col = None
    for candidate in ("volume_uc", "volume", "volume_sellout"):
        if candidate in grp.columns:
            vol_col = candidate
            break
    if vol_col and vol_col != "volume_uc":
        grp = grp.rename(columns={vol_col: "volume_uc"})
    if "volume_uc" not in grp.columns:
        grp["volume_uc"] = 0.0
    for c in ("volume_uc", "receita_liquida", "resultado_operacional"):
        if c not in grp.columns:
            grp[c] = 0.0
        grp[c] = pd.to_numeric(grp[c], errors="coerce").fillna(0.0)
    if scen_col not in grp.columns:
        grp[scen_col] = "(sem_cenario)"
    out = grp.groupby(scen_col, dropna=False)[["volume_uc", "receita_liquida", "resultado_operacional"]].sum().reset_index()
    out = out.rename(columns={scen_col: "cenario"})
    return out.sort_values("cenario")

def print_current_summary(df_current: pd.DataFrame, logger: Optional[logging.Logger] = None) -> None:
    log = logger or setup_logger("etl.summary")
    if df_current is None or df_current.empty:
        log.info("current.parquet: [vazio]")
        return
    anos = extract_years(df_current)
    scen_col = scenario_column(df_current)
    if scen_col:
        cenarios = sorted(set(df_current[scen_col].dropna().astype(str).str.strip().tolist()))
    else:
        cenarios = []
    log.info("=== Resumo por Cenário (current.parquet) ===")
    if cenarios:
        log.info(f"Cenários distintos: {len(cenarios)} → amostra: {cenarios[:10]}")
    if anos:
        log.info(f"Períodos (anos): {anos}")
    resumo = summarize_current_by_cenario(df_current)
    if resumo.empty:
        log.info("(sem linhas)")
        return
    header = f"{'Cenário':30} {'Volume_UC':>16} {'Receita_Líquida':>20} {'Resultado_Operacional':>24}"
    log.info(header)
    log.info("-" * len(header))
    for _, r in resumo.iterrows():
        log.info(f"{str(r['cenario'])[:30]:30} "
                 f"{_fmt_brl(r['volume_uc']):>16} "
                 f"{_fmt_brl(r['receita_liquida']):>20} "
                 f"{_fmt_brl(r['resultado_operacional']):>24}")
    tot = resumo[["volume_uc","receita_liquida","resultado_operacional"]].sum()
    log.info("-" * len(header))
    log.info(f"{'Total':30} "
             f"{_fmt_brl(tot['volume_uc']):>16} "
             f"{_fmt_brl(tot['receita_liquida']):>20} "
             f"{_fmt_brl(tot['resultado_operacional']):>24}")

def print_res_working_summary(df_res: pd.DataFrame, logger: Optional[logging.Logger] = None) -> None:
    log = logger or setup_logger("etl.summary")
    if df_res is None or df_res.empty:
        log.info("res_working.parquet: [vazio]")
        return
    anos = extract_years(df_res)
    fam_com_col = find_column(df_res, ["Família Comercial", "Familia Comercial"])
    fams = sorted(set(df_res[fam_com_col].dropna().astype(str).str.strip().tolist())) if fam_com_col else []
    vol = pd.to_numeric(df_res.get("volume", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
    log.info("=== Resumo RE (res_working.parquet) ===")
    if anos:
        log.info(f"Anos: {anos}")
    if fams:
        log.info(f"Famílias: {len(fams)}")
    log.info(f"Volume total (volume): {_fmt_brl(vol)}")  # esse é o código original


# =============================================================================
# TECNOLOGIA — CARREGAMENTO E APLICAÇÃO (ADICIONADO)
# =============================================================================

def _normalize_tecnologia(val) -> Optional[str]:
    """Normaliza rótulo de Tecnologia (pode customizar: Title Case, trim, etc.)."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    return s  # manter como veio; ajuste se quiser padronizar caixa


def load_tecnologia_maps(
    base_dir: Path = DATA_DIR,
    subdir: str = "tec_poh",
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, int]]:
    """
    Lê TODOS os .xlsx em data/<subdir>/ (default: data/tec_poh/) e produz:
      - sku_to_tecnologia: dict SKU -> Tecnologia  (ABSOLUTO; modo/mais frequente em caso de conflito)
      - familia_to_tecnologia: dict Família Comercial -> Tecnologia (fallback; modo/mais frequente)
      - stats: contadores básicos (linhas, skus únicos, famílias únicas, conflitos)
    Regras:
      - SKU sempre texto (normalize_sku_text)
      - Colunas detectadas por find_column:
        * SKU (opcional, mas necessário para map por SKU)
        * Família Comercial (opcional, melhora fallback)
        * Tecnologia (obrigatório)
    """
    log = logger or setup_logger("etl.common")

    src_dir = Path(base_dir) / subdir
    try:
        _assert_path_under_data(src_dir, log)
    except Exception:
        pass

    files = sorted([*src_dir.glob("*.xlsx")])
    if not files:
        log.warning(f"[TECNO] Nenhum .xlsx encontrado em {src_dir}.")
        return {}, {}, {"rows": 0, "skus": 0, "familias": 0, "conf_sku": 0, "conf_fam": 0}

    frames = []
    for f in files:
        try:
            df = pd.read_excel(f, dtype=str)
        except Exception:
            df = pd.read_excel(f)
        df.columns = [str(c).strip() for c in df.columns]
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)

    sku_col = find_column(raw, [
        "SKU","sku","Sku",
        "Cód. Produto","Cod. Produto","Código do Produto","Codigo do Produto",
        "Código Produto","Codigo Produto","Cod Produto"
    ])
    fam_col = find_column(raw, ["Família Comercial", "Familia Comercial"])
    tec_col = find_column(raw, ["Tecnologia", "TECNOLOGIA", "tecnologia"])

    if tec_col is None:
        log.warning("[TECNO] Coluna 'Tecnologia' não encontrada na referência.")
        return {}, {}, {"rows": 0, "skus": 0, "familias": 0, "conf_sku": 0, "conf_fam": 0}

    if sku_col and sku_col != "SKU":
        raw = raw.rename(columns={sku_col: "SKU"})
    if fam_col and fam_col != "Família Comercial":
        raw = raw.rename(columns={fam_col: "Família Comercial"})
    if tec_col != "Tecnologia":
        raw = raw.rename(columns={tec_col: "Tecnologia"})

    if "SKU" in raw.columns:
        raw["SKU"] = normalize_sku_text(raw["SKU"])
    if "Família Comercial" in raw.columns:
        raw["Família Comercial"] = raw["Família Comercial"].astype(str).str.strip()
    raw["Tecnologia"] = raw["Tecnologia"].map(_normalize_tecnologia)

    # Map SKU → Tecnologia (modo)
    sku_to_tecnologia: Dict[str, str] = {}
    conf_sku = 0
    if "SKU" in raw.columns:
        tmp = raw.dropna(subset=["SKU", "Tecnologia"]).copy()
        if not tmp.empty:
            grp = tmp.groupby("SKU")["Tecnologia"].agg(
                lambda x: x.mode().iat[0] if not x.mode().empty else x.dropna().iat[0]
            )
            cnt = tmp.groupby("SKU")["Tecnologia"].nunique()
            conf_sku = int((cnt > 1).sum())
            sku_to_tecnologia = grp.to_dict()

    # Map Família → Tecnologia (modo)
    familia_to_tecnologia: Dict[str, str] = {}
    conf_fam = 0
    if "Família Comercial" in raw.columns:
        tmpf = raw.dropna(subset=["Família Comercial", "Tecnologia"]).copy()
        if not tmpf.empty:
            grpf = tmpf.groupby("Família Comercial")["Tecnologia"].agg(
                lambda x: x.mode().iat[0] if not x.mode().empty else x.dropna().iat[0]
            )
            cntf = tmpf.groupby("Família Comercial")["Tecnologia"].nunique()
            conf_fam = int((cntf > 1).sum())
            familia_to_tecnologia = grpf.to_dict()

    stats = {
        "rows": int(len(raw)),
        "skus": int(raw["SKU"].nunique()) if "SKU" in raw.columns else 0,
        "familias": int(raw["Família Comercial"].nunique()) if "Família Comercial" in raw.columns else 0,
        "conf_sku": conf_sku,
        "conf_fam": conf_fam,
    }

    log.info(f"[TECNO] Carregado de {src_dir} | linhas={stats['rows']} | skus={stats['skus']} | "
             f"familias={stats['familias']} | conflitos SKU={stats['conf_sku']} | conflitos Família={stats['conf_fam']}")
    if stats["conf_sku"] > 0:
        log.warning("[TECNO] Há SKU(s) com múltiplas tecnologias — deduplicar referência ou validar regra (modo aplicado).")
    if stats["conf_fam"] > 0:
        log.warning("[TECNO] Há Família(s) com múltiplas tecnologias — fallback por família pode ficar ambíguo (modo aplicado).")

    return sku_to_tecnologia, familia_to_tecnologia, stats


def apply_tecnologia(
    df: pd.DataFrame,
    *,
    base_dir: Optional[Path] = None,
    sku_map: Optional[Dict[str, str]] = None,
    fam_map: Optional[Dict[str, str]] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Enriquecimento de 'Tecnologia' no DF:
      1) ABSOLUTO por SKU (se existir coluna 'SKU' e sku_map)
      2) Fallback por 'Família Comercial' (se existir e fam_map)
    - Não remove colunas; cria/preenche 'Tecnologia' quando ausente.
    - Se sku_map/fam_map não forem passados, carrega de data/tec_poh via load_tecnologia_maps(base_dir=...).
    """
    log = logger or setup_logger("etl.common")
    out = df.copy()

    if "Tecnologia" not in out.columns:
        out["Tecnologia"] = np.nan

    # Carregar mapas se necessário
    if sku_map is None and fam_map is None:
        if base_dir is None:
            base_dir = DATA_DIR
        sku_map, fam_map, _ = load_tecnologia_maps(base_dir=base_dir, logger=log)

    # 1) Por SKU (absoluto)
    if "SKU" in out.columns and sku_map:
        # garantir normalização de SKU
        out["__SKU_norm__"] = normalize_sku_text(out["SKU"])
        out["Tecnologia"] = out["Tecnologia"].fillna(out["__SKU_norm__"].map(lambda s: sku_map.get(s, np.nan)))
        out = out.drop(columns=["__SKU_norm__"], errors="ignore")

    # 2) Fallback por Família
    fam_col = find_column(out, ["Família Comercial", "Familia Comercial"])
    if fam_col:
        if fam_col != "Família Comercial":
            out = out.rename(columns={fam_col: "Família Comercial"})
        if fam_map:
            out["Tecnologia"] = out["Tecnologia"].fillna(
                out["Família Comercial"].astype(str).str.strip().map(lambda f: fam_map.get(f, np.nan))
            )

    # Sanitização final
    out["Tecnologia"] = out["Tecnologia"].astype("string")

    # Telemetria
    filled = int(out["Tecnologia"].notna().sum())
    log.info(f"[TECNO] Tecnologia preenchida em {filled}/{len(out)} linha(s) ({(filled/len(out))*100:.1f}%).")

    return out
