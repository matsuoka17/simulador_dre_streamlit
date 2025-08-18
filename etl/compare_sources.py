# -*- coding: utf-8 -*-
"""
etl/compare_sources.py
Auditoria Parquet vs Excel (largo) por cenário, mês e linhas do P&L.

- Lê 'data/current.parquet' (saída do ETL) e o Excel "largo" original.
- Usa o CENÁRIO **literal** do Excel (sem normalização):
    cenario_label = texto da coluna "Cenário" (strip)
    cenario       = cenario_label
- Remove linhas de subtotal/ALL do Excel antes de agregar.
- Agrega no mesmo grão do parquet: ano, mes, indicador_id, cenario, cenario_label.
- Compara valores, reporta diferenças e salva CSVs de diagnóstico em data/debug_compare/.

Uso:
    python etl/compare_sources.py
    python etl/compare_sources.py --parquet data/current.parquet --excel etl/realizado/raw/arquivo.xls
"""

from __future__ import annotations
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# -------------------- Config logging --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("compare")

# -------------------- Localização padrão --------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../simulador_dre_streamlit
PARQUET_DEFAULT = PROJECT_ROOT / "data" / "current.parquet"
EXCEL_DIR_DEFAULT = PROJECT_ROOT / "etl" / "realizado" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "debug_compare"

# -------------------- Mapeamentos (mesmo mapa do ETL) --------------------
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
MES_MAP = {1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
           7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"}

# -------------------- Utilitários --------------------
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
    # tenta data AAAA-MM-DD, etc.
    dt = pd.to_datetime(s, errors="coerce")
    if pd.notna(dt):
        mm = int(dt.month)
        return mm if 1 <= mm <= 12 else None
    return None

def _infer_ano_from_cenario_literal(cenario_label: str) -> Optional[int]:
    """Se não houver coluna 'Ano', tenta inferir do texto literal do Cenário."""
    if not isinstance(cenario_label, str):
        return None
    s = cenario_label.strip()
    # AAAA
    m = pd.Series(s).str.extract(r"(\d{4})").iloc[0, 0]
    if pd.notna(m):
        return int(m)
    # Dois dígitos no fim (Realizado 25 -> 2025)
    m2 = pd.Series(s).str.extract(r"(\d{2})\s*$").iloc[0, 0]
    if pd.notna(m2):
        yy = int(m2)
        return 2000 + yy if yy < 70 else 1900 + yy
    return None

def _find_latest_excel(directory: Path) -> Optional[Path]:
    exts = (".xlsx", ".xls", ".csv")
    files = [p for p in directory.glob("*") if p.suffix.lower() in exts and not p.name.startswith("~$")]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)

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
        log.info(f"Removendo {removed} linha(s) de subtotal/ALL no Excel.")
    return df.loc[~mask].copy()

# -------------------- Carga e preparação --------------------
def load_parquet_agg(parquet_path: Path) -> pd.DataFrame:
    dfp = pd.read_parquet(parquet_path)
    dfp["ano"] = pd.to_numeric(dfp["ano"], errors="coerce").astype("Int64")
    dfp["mes"] = pd.to_numeric(dfp["mes"], errors="coerce").astype("Int64")
    dfp["valor"] = pd.to_numeric(dfp["valor"], errors="coerce").fillna(0.0)
    dfp["indicador_id"] = dfp["indicador_id"].astype(str)
    dfp["cenario_label"] = dfp["cenario_label"].astype(str).str.strip()
    dfp["cenario"] = dfp["cenario"].astype(str).str.strip()

    keys = ["ano", "mes", "indicador_id", "cenario", "cenario_label"]
    dfp_agg = dfp.groupby(keys, dropna=False, as_index=False)["valor"].sum()
    dfp_agg = dfp_agg.rename(columns={"valor": "valor_parquet"})
    return dfp_agg

def load_excel_agg(excel_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(excel_path) if excel_path.suffix.lower() == ".csv" else pd.read_excel(excel_path)
    raw = _drop_excel_subtotals(raw)

    # Descobre col de mês e ano
    mes_col = "Mês" if "Mês" in raw.columns else ("Mes" if "Mes" in raw.columns else None)
    ano_col = "Ano" if "Ano" in raw.columns else None
    has_cenario = "Cenário" in raw.columns

    # Quais indicadores existem no Excel?
    indicator_cols = [c for c in INDICATOR_MAP.keys() if c in raw.columns]
    if not indicator_cols:
        raise ValueError("Nenhuma coluna de indicador conhecida foi encontrada no Excel.")

    df = raw.copy()

    # Mês -> número
    if mes_col:
        df["mes"] = df[mes_col].apply(_normalize_mes).astype("Int64")
    else:
        df["mes"] = pd.NA

    # Ano
    if ano_col:
        df["ano"] = pd.to_numeric(df[ano_col], errors="coerce").astype("Int64")
    else:
        df["ano"] = df["Cenário"].apply(_infer_ano_from_cenario_literal).astype("Int64") if has_cenario else pd.NA

    # Melt largo → longo (somente indicadores)
    id_vars = [c for c in ["Cenário", "mes", "ano"] if c in df.columns]
    melted = df[id_vars + indicator_cols].melt(
        id_vars=id_vars,
        value_vars=indicator_cols,
        var_name="indicador_nome",
        value_name="valor"
    )

    # Tipagem
    melted["valor"] = pd.to_numeric(melted["valor"], errors="coerce").fillna(0.0)

    # Indicador → id canônico
    melted["indicador_id"] = melted["indicador_nome"].map(INDICATOR_MAP)
    melted.dropna(subset=["indicador_id"], inplace=True)

    # >>> Cenário literal (sem normalização) <<<
    if has_cenario:
        melted["cenario_label"] = melted["Cenário"].astype(str).str.strip()
        melted["cenario"] = melted["cenario_label"]
    else:
        melted["cenario_label"] = "Realizado"
        melted["cenario"] = "Realizado"

    # Agrega no mesmo grão do parquet
    keys = ["ano", "mes", "indicador_id", "cenario", "cenario_label"]
    dfx_agg = (melted[keys + ["valor"]]
               .groupby(keys, dropna=False, as_index=False)["valor"].sum())
    dfx_agg = dfx_agg.rename(columns={"valor": "valor_excel"})
    return dfx_agg

# -------------------- Comparação --------------------
def compare_frames(dfp: pd.DataFrame, dfx: pd.DataFrame) -> pd.DataFrame:
    keys = ["ano", "mes", "indicador_id", "cenario", "cenario_label"]
    comp = dfx.merge(dfp, on=keys, how="outer")
    comp["valor_excel"] = pd.to_numeric(comp["valor_excel"], errors="coerce").fillna(0.0)
    comp["valor_parquet"] = pd.to_numeric(comp["valor_parquet"], errors="coerce").fillna(0.0)

    comp["diff"] = comp["valor_parquet"] - comp["valor_excel"]
    comp["diff_abs"] = comp["diff"].abs()
    comp["diff_pct"] = np.where(comp["valor_excel"].abs() > 1e-12,
                                comp["diff"] / comp["valor_excel"],
                                np.nan)
    comp["mes_nome"] = comp["mes"].map(MES_MAP)

    comp = comp.sort_values(["ano", "cenario", "diff_abs"], ascending=[True, True, False]).reset_index(drop=True)
    return comp

def summarize(comp: pd.DataFrame) -> pd.DataFrame:
    return (comp.groupby(["indicador_id", "cenario"], as_index=False)["diff_abs"].sum()
                .sort_values("diff_abs", ascending=False))

def summarize_by_month(comp: pd.DataFrame) -> pd.DataFrame:
    return (comp.groupby(["mes", "mes_nome", "cenario"], as_index=False)["diff_abs"].sum()
                .sort_values(["cenario", "mes"], ascending=[True, True]))

def pivot_matrix(comp: pd.DataFrame) -> pd.DataFrame:
    mat = (comp.pivot_table(index="indicador_id", columns="mes_nome", values="diff", aggfunc="sum", fill_value=0.0)
               .reindex(columns=[MES_MAP[m] for m in range(1, 13)], fill_value=0.0))
    mat["Total"] = mat.sum(axis=1)
    return mat.reset_index()

# -------------------- Main --------------------
def _find_latest_excel_file() -> Path:
    p = _find_latest_excel(EXCEL_DIR_DEFAULT)
    if not p:
        raise FileNotFoundError(f"Nenhum Excel encontrado em {EXCEL_DIR_DEFAULT}")
    return p

def main(parquet_path: Optional[str], excel_path: Optional[str]):
    parquet_p = Path(parquet_path) if parquet_path else PARQUET_DEFAULT
    excel_p = Path(excel_path) if excel_path else _find_latest_excel_file()

    log.info(f"PARQUET : {parquet_p}")
    log.info(f"EXCEL   : {excel_p}")

    if not parquet_p.exists():
        raise FileNotFoundError(f"Parquet não encontrado: {parquet_p}")
    if not excel_p.exists():
        raise FileNotFoundError(f"Excel não encontrado: {excel_p}")

    dfp = load_parquet_agg(parquet_p)
    dfx = load_excel_agg(excel_p)

    # Interseção de anos para reduzir ruído
    anos = sorted(set(dfp["ano"].dropna().unique()).intersection(set(dfx["ano"].dropna().unique())))
    if anos:
        dfp = dfp[dfp["ano"].isin(anos)]
        dfx = dfx[dfx["ano"].isin(anos)]

    comp = compare_frames(dfp, dfx)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = OUT_DIR / f"{ts}_full_diffs.csv"
    comp.to_csv(full_path, index=False, encoding="utf-8-sig")

    top_ind = summarize(comp)
    top_ind_path = OUT_DIR / f"{ts}_top_diffs_by_indicator.csv"
    top_ind.to_csv(top_ind_path, index=False, encoding="utf-8-sig")

    top_month = summarize_by_month(comp)
    top_month_path = OUT_DIR / f"{ts}_top_diffs_by_month.csv"
    top_month.to_csv(top_month_path, index=False, encoding="utf-8-sig")

    mat = pivot_matrix(comp)
    mat_path = OUT_DIR / f"{ts}_pivot_diff_matrix.csv"
    mat.to_csv(mat_path, index=False, encoding="utf-8-sig")

    total_abs = comp["diff_abs"].sum()
    log.info(f"Diferença absoluta total: {total_abs:,.2f}".replace(",", "§").replace(".", ",").replace("§", "."))
    log.info(f"[OK] Gerados:\n- {full_path}\n- {top_ind_path}\n- {top_month_path}\n- {mat_path}")

    preview_cols = ["ano", "cenario", "cenario_label", "mes_nome", "indicador_id",
                    "valor_excel", "valor_parquet", "diff", "diff_pct"]
    preview = comp.head(10)[[c for c in preview_cols if c in comp.columns]]
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("\nTop 10 diferenças (amostra):")
        print(preview.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparar Parquet vs Excel (largo) por cenário/mês/indicador.")
    parser.add_argument("--parquet", type=str, default=None, help="Caminho do parquet (default: data/current.parquet)")
    parser.add_argument("--excel", type=str, default=None, help="Caminho do Excel largo (default: mais recente em etl/realizado/raw)")
    args = parser.parse_args()
    main(args.parquet, args.excel)
