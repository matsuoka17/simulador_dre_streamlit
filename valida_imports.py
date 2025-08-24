# valida_imports.py
# -*- coding: utf-8 -*-
"""
Valida e projeta P&L a partir dos Parquets gerados pela ETL.
- Lê current.parquet (realizado) e res_working.parquet (metadata/comparativos)
- Mostra P&L consolidado (todas as linhas do DRE disponíveis),
  e GERA MATRIZ com MESES NAS COLUNAS e LINHAS DO P&L NAS LINHAS.
- Na aba de famílias, inclui a coluna Tecnologia.
- Salva Excel com abas (dre_realizado.xlsx) no mesmo diretório dos Parquets.

Uso:
  python valida_imports.py --parquet-dir C:\...\simulador_dre_streamlit\data\parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# Ordem preferencial das linhas do DRE (quando existirem nas colunas)
_PNL_PREF_ORDER = [
    # Volumes
    "volume_uc", "volume_sellout",
    # Receita e deduções
    "receita_bruta", "convenio_impostos", "receita_liquida",
    # Custos / variáveis
    "insumos", "custos_variaveis", "estrutura_industrial_variavel", "margem_variavel",
    # Estrutura fixa e margem bruta
    "estrutura_industrial_fixa", "margem_bruta",
    # Logística e despesas
    "custos_toll_packer", "fretes_t1", "custos_log_t1_reg",
    "despesas_secundarias_t2", "perdas",
    # Contribuição
    "margem_contribuicao", "dme", "margem_contribuicao_liquida",
    # Opex / Depre / Resultado / EBITDA
    "opex_leao_reg", "depreciacao", "resultado_operacional", "ebitda",
]

# Dimensões padrão
_DIMS_CANON = ["SKU", "Família Comercial", "produto_descricao", "sistema", "cenario", "ano", "mes"]


def parse_args():
    p = argparse.ArgumentParser()
    default_dir = Path(__file__).resolve().parent / "data" / "parquet"
    p.add_argument("--parquet-dir", type=str, default=str(default_dir),
                   help="Diretório contendo current.parquet e res_working.parquet")
    p.add_argument("--cenario-like", type=str, default="Realizado",
                   help="Filtro de cenário (substring, case-insensitive). Ex.: 'Realizado'")
    return p.parse_args()


def check_exists(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(
            f"{name} não encontrado em:\n  {path}\n"
            "Verifique se a ETL gerou na pasta correta ou informe --parquet-dir."
        )


def order_metrics(metrics: list[str]) -> list[str]:
    """Ordena as métricas conforme a ordem preferencial; o que não entrar vai ao final (ordem alfabética)."""
    pref = [m for m in _PNL_PREF_ORDER if m in metrics]
    rest = sorted([m for m in metrics if m not in pref])
    return pref + rest


def detect_dims_metrics(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    dims = [c for c in _DIMS_CANON if c in df.columns]
    metrics = [c for c in df.columns if c not in dims and pd.api.types.is_numeric_dtype(df[c])]
    return dims, order_metrics(metrics)


def add_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona KPIs usuais quando as colunas existem."""
    df = df.copy()
    # Margem Operacional (%)
    if {"receita_liquida", "resultado_operacional"}.issubset(df.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            df["margem_op_%"] = np.where(
                df["receita_liquida"] != 0,
                (df["resultado_operacional"] / df["receita_liquida"]) * 100.0,
                np.nan,
            )
    # Receita por UC
    if {"receita_liquida", "volume_uc"}.issubset(df.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            df["receita_por_uc"] = np.where(
                df["volume_uc"] != 0,
                df["receita_liquida"] / df["volume_uc"],
                np.nan,
            )
    # Margem EBITDA (%)
    if {"receita_liquida", "ebitda"}.issubset(df.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            df["margem_ebitda_%"] = np.where(
                df["receita_liquida"] != 0,
                (df["ebitda"] / df["receita_liquida"]) * 100.0,
                np.nan,
            )
    return df


def build_pnl_matrix_months(df_real: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """
    Constrói MATRIZ P&L: linhas = métricas (linhas do DRE) | colunas = meses (YYYY-MM).
    """
    if not {"ano", "mes"}.issubset(df_real.columns):
        return pd.DataFrame()

    # Agrega por ano/mês para todas as métricas
    g = (df_real.groupby(["ano", "mes"], dropna=False)[metrics]
         .sum()
         .reset_index())

    # Label de coluna YYYY-MM e ordenação temporal
    g = g.dropna(subset=["ano", "mes"]).copy()
    g["mes"] = g["mes"].astype(int)
    g["ano"] = g["ano"].astype(int)
    g["periodo"] = g["ano"].astype(str) + "-" + g["mes"].astype(str).str.zfill(2)

    # Long -> pivot (métrica na linha, período na coluna)
    long = g.melt(id_vars=["periodo"], value_vars=metrics, var_name="linha_pnl", value_name="valor")
    # Ordena linhas pela ordem preferencial
    order = order_metrics(metrics)
    long["linha_pnl"] = pd.Categorical(long["linha_pnl"], categories=order, ordered=True)

    mat = (long.pivot_table(index="linha_pnl", columns="periodo", values="valor", aggfunc="sum")
           .sort_index())

    # Ordenar colunas por período (YYYY-MM)
    try:
        mat = mat.reindex(sorted(mat.columns, key=lambda s: (int(s.split("-")[0]), int(s.split("-")[1]))), axis=1)
    except Exception:
        mat = mat  # mantém como está se falhar parsing

    # Coluna Total (soma do ano inteiro / períodos disponíveis)
    mat["Total"] = mat.sum(axis=1)
    return mat.reset_index()


def main():
    args = parse_args()
    parquet_dir = Path(args.parquet_dir).resolve()
    cur_path = parquet_dir / "current.parquet"
    res_path = parquet_dir / "res_working.parquet"
    out_xlsx = parquet_dir / "dre_realizado.xlsx"

    print(f"[INFO] Parquet dir: {parquet_dir}")
    check_exists(cur_path, "current.parquet")
    res_exists = res_path.exists()

    # Requer pyarrow/fastparquet
    df_cur = pd.read_parquet(cur_path)
    df_res = pd.read_parquet(res_path) if res_exists else None

    # Filtro de cenário
    if "cenario" in df_cur.columns:
        mask = df_cur["cenario"].astype(str).str.contains(args.cenario_like, case=False, na=False)
        df_real = df_cur.loc[mask].copy()
        cenario_usado = ", ".join(sorted(df_cur.loc[mask, "cenario"].dropna().unique())) or "(vazio)"
    else:
        df_real = df_cur.copy()
        cenario_usado = "(sem coluna cenario)"

    # Detecta dims x métricas dinamicamente
    dims, metrics = detect_dims_metrics(df_real)
    if not metrics:
        raise RuntimeError("Não encontrei colunas numéricas de P&L no current.parquet.")

    # =============================
    # P&L TOTAL (todas as métricas)
    # =============================
    pl_total = df_real[metrics].sum().to_frame(name="total").T
    pl_total = add_kpis(pl_total)

    # ============================================================
    # MATRIZ P&L (MESES NAS COLUNAS | LINHAS = MÉTRICAS DO DRE)
    # ============================================================
    pnl_matrix = build_pnl_matrix_months(df_real, metrics)

    # ============================================================
    # P&L por Família Comercial (+ Tecnologia), agregando métricas
    # ============================================================
    fam_col = "Família Comercial" if "Família Comercial" in df_real.columns else None
    tec_col = "Tecnologia" if "Tecnologia" in df_real.columns else None
    if fam_col:
        group_dims = [fam_col] + ([tec_col] if tec_col else [])
        sort_col = "receita_liquida" if "receita_liquida" in metrics else metrics[0]
        pl_fam = (df_real.groupby(group_dims, dropna=False)[metrics]
                  .sum()
                  .sort_values(sort_col, ascending=False)
                  .reset_index())
        pl_fam = add_kpis(pl_fam)
    else:
        pl_fam = pd.DataFrame()

    # ==========================
    # Impressão executiva (CLI)
    # ==========================
    print("\n=== P&L REALIZADO — cenário:", cenario_usado, "===")
    cols_print = order_metrics([c for c in pl_total.columns if c in metrics])
    for extra in ["margem_op_%", "margem_ebitda_%", "receita_por_uc"]:
        if extra in pl_total.columns:
            cols_print.append(extra)
    print(pl_total[cols_print].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    if not pnl_matrix.empty:
        # Mostra só últimos 12 períodos
        all_periods = [c for c in pnl_matrix.columns if c not in ("linha_pnl", "Total")]
        tail_periods = sorted(all_periods)[-12:]
        cols_show = ["linha_pnl"] + tail_periods + (["Total"] if "Total" in pnl_matrix.columns else [])
        print("\n=== MATRIZ P&L (últimos 12 períodos) ===")
        print(pnl_matrix[cols_show].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    if not pl_fam.empty:
        print("\n=== Top 15 Famílias por Receita Líquida ===")
        order_cols = ([fam_col] + ([tec_col] if tec_col else []) + cols_print)
        key = "receita_liquida" if "receita_liquida" in pl_fam.columns else cols_print[0]
        print(pl_fam.nlargest(15, key)[order_cols]
              .to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # ==============================
    # Exporta Excel com todas abas
    # ==============================
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        meta_rows = [{"recurso": "current.parquet", "linhas": len(df_cur), "colunas": len(df_cur.columns),
                      "cenario_utilizado": cenario_usado}]
        if df_res is not None:
            meta_rows.append({"recurso": "res_working.parquet", "linhas": len(df_res), "colunas": len(df_res.columns),
                              "cenario_utilizado": ""})
        pd.DataFrame(meta_rows).to_excel(writer, sheet_name="metadata", index=False)

        # Total consolidado
        pl_total.to_excel(writer, sheet_name="pl_total", index=False)

        # MATRIZ por meses: meses nas colunas, linhas do P&L nas linhas
        if not pnl_matrix.empty:
            pnl_matrix.to_excel(writer, sheet_name="pl_ano_mes", index=False)

        # Famílias (+Tecnologia)
        if not pl_fam.empty:
            # Garante presença da coluna Tecnologia, mesmo que vazia (para contrato claro)
            if "Tecnologia" not in pl_fam.columns:
                pl_fam["Tecnologia"] = pd.NA
            # Reordena para exibir dimensões primeiro
            dim_cols = ["Família Comercial", "Tecnologia"]
            dim_cols = [c for c in dim_cols if c in pl_fam.columns]
            metric_cols = [c for c in pl_fam.columns if c not in dim_cols and pd.api.types.is_numeric_dtype(pl_fam[c])]
            kpi_cols = [c for c in ["margem_op_%", "margem_ebitda_%", "receita_por_uc"] if c in pl_fam.columns]
            ordered_cols = dim_cols + order_metrics(metric_cols) + kpi_cols
            pl_fam[ordered_cols].to_excel(writer, sheet_name="pl_familia", index=False)

    print(f"\n[OK] Excel gerado: {out_xlsx}")


if __name__ == "__main__":
    main()
