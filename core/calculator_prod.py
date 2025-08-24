# core/calculo_poh.py
# --------------------------------------------------------------------------------------
# POH — Cálculo/Loaders
# - Carrega:
#     * data/parquet/current.parquet
#     * data/parquet/res_working.parquet
#     * data/cap_poh/capacidade_tecnologias* (xlsx/csv/parquet)
# - Entrega:
#     * load_capacidade_tecnologias()            -> DF long [Tecnologia, ano, mes, capacidade]
#     * familia_tecnologia_map(current_df, ano)  -> DF [Família Comercial, Tecnologia]
#     * compute_cutoff_for_year(current_df, ano) -> int (último mês YTD)
#     * build_demanda_ytg(res_df, map_df, ano, cutoff) -> DF [Tecnologia, ano, mes, demanda]
#     * build_cap_vs_demanda(ano, ...)           -> DF Tec×Mês com Cap, Dem, Gap, Crítico
# - Placeholder opcional:
#     * simular_rolagem(...)  -> skeleton (não usado pela página 04_POH)
# --------------------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import logging

# ------------------------------------------------------------------------------
# Paths (sempre relativos à raiz do projeto)
# ------------------------------------------------------------------------------
DATA_DIR = Path("data")
PARQUET_DIR = DATA_DIR / "parquet"
CAP_DIR = DATA_DIR / "cap_poh"
ESTOQUE_INICIAL_DIR = DATA_DIR / "estoque" / "inicial"
POLITICA_ESTOQUE_DIR = DATA_DIR / "estoque" / "politica"

CURRENT_PARQUET = PARQUET_DIR / "current.parquet"
RES_WORKING_PARQUET = PARQUET_DIR / "res_working.parquet"

# ------------------------------------------------------------------------------
# Constantes auxiliares
# ------------------------------------------------------------------------------
MONTHS_PT = {
    1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"
}
MESES = list(range(1, 13))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("calculo_poh")

# ------------------------------------------------------------------------------
# Utilidades de IO
# ------------------------------------------------------------------------------
def _read_parquet_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)

def _latest_matching(base_dir: Path, glob_pattern: str) -> Optional[Path]:
    cands = sorted(
        base_dir.glob(glob_pattern),
        key=lambda p: (p.stat().st_mtime if p.exists() else 0),
        reverse=True
    )
    return cands[0] if cands else None

# ------------------------------------------------------------------------------
# Carregadores básicos (current/res_working)
# ------------------------------------------------------------------------------
def load_current() -> pd.DataFrame:
    """
    Lê current.parquet. Esperado conter:
      - chaves: cenario, ano, mes, indicador_id, Família Comercial [, Tecnologia]
      - colunas: valor (float), volume_uc (int/float)
    """
    df = _read_parquet_safe(CURRENT_PARQUET)
    if "mes" in df.columns:
        df["mes"] = pd.to_numeric(df["mes"], errors="coerce").fillna(0).astype(int)
    if "ano" in df.columns:
        df["ano"] = pd.to_numeric(df["ano"], errors="coerce").fillna(0).astype(int)
    # coerção leve
    if "volume_uc" in df.columns:
        df["volume_uc"] = pd.to_numeric(df["volume_uc"], errors="coerce").fillna(0.0)
    if "valor" in df.columns:
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce").fillna(0.0)
    return df

def load_res_working() -> pd.DataFrame:
    """
    Lê res_working.parquet (volumes UI/RES do YTG por família/mês).
    Normaliza coluna de volume para 'volume_uc'.
    """
    df = _read_parquet_safe(RES_WORKING_PARQUET)
    if "mes" in df.columns:
        df["mes"] = pd.to_numeric(df["mes"], errors="coerce").fillna(0).astype(int)
    if "ano" in df.columns:
        df["ano"] = pd.to_numeric(df["ano"], errors="coerce").fillna(0).astype(int)
    vol_col = None
    if "volume_uc" in df.columns:
        vol_col = "volume_uc"
    else:
        for c in df.columns:
            if str(c).lower().strip() in {"volume", "vol_uc", "volumes", "vol"}:
                vol_col = c
                break
    if vol_col:
        df["volume_uc"] = pd.to_numeric(df[vol_col], errors="coerce").fillna(0.0)
        if vol_col != "volume_uc":
            df = df.rename(columns={vol_col: "volume_uc"})
    else:
        df["volume_uc"] = 0.0
    return df

# ------------------------------------------------------------------------------
# Loader de capacidade (robusto: long / largo Jan..Dez / constante)
# ------------------------------------------------------------------------------
def load_capacidade_tecnologias() -> pd.DataFrame:
    """
    Retorna DF long com colunas: ['Tecnologia','ano','mes','capacidade'].
    Procura o arquivo mais recente em data/cap_poh/ que case com 'capacidade_tecnologias*.*'.
    Formatos aceitos:
      (A) long:  tecnologia, ano, mes, capacidade
      (B) largo: tecnologia, ano, Jan..Dez  (melt)
      (C) constante: tecnologia, capacidade  (replica 12 meses no ano corrente)
    """
    path = _latest_matching(CAP_DIR, "capacidade_tecnologias*.*")
    if path is None:
        log.info("[POH] Capacidade: arquivo não encontrado em %s", CAP_DIR)
        return pd.DataFrame(columns=["Tecnologia", "ano", "mes", "capacidade"])

    # Leitura
    ext = path.suffix.lower()
    if ext in {".xlsx", ".xls"}:
        raw = pd.read_excel(path)
    elif ext in {".csv"}:
        raw = pd.read_csv(path)
    elif ext in {".parquet", ".pq", ".parq"}:
        raw = pd.read_parquet(path)
    else:
        # fallback tentativo
        raw = pd.read_excel(path)

    cols = [str(c).strip().lower() for c in raw.columns]
    raw.columns = cols

    # Caso A — long
    if {"tecnologia", "ano", "mes", "capacidade"}.issubset(cols):
        df = raw[["tecnologia", "ano", "mes", "capacidade"]].copy()
        df["ano"] = pd.to_numeric(df["ano"], errors="coerce").fillna(0).astype(int)
        df["mes"] = pd.to_numeric(df["mes"], errors="coerce").fillna(0).astype(int)
        df["capacidade"] = pd.to_numeric(df["capacidade"], errors="coerce").fillna(0.0)
        return df.rename(columns={"tecnologia": "Tecnologia"})

    # Caso B — largo Jan..Dez
    meses_map = {"jan":1,"fev":2,"mar":3,"abr":4,"mai":5,"jun":6,"jul":7,"ago":8,"set":9,"out":10,"nov":11,"dez":12}
    if {"tecnologia","ano"}.issubset(cols):
        mes_cols = [c for c in cols if c[:3] in meses_map]
        if mes_cols:
            df = raw[["tecnologia", "ano"] + mes_cols].copy()
            df = df.melt(id_vars=["tecnologia","ano"], var_name="mes_nome", value_name="capacidade")
            df["mes"] = df["mes_nome"].str[:3].map(meses_map).astype(int)
            df["ano"] = pd.to_numeric(df["ano"], errors="coerce").fillna(0).astype(int)
            df["capacidade"] = pd.to_numeric(df["capacidade"], errors="coerce").fillna(0.0)
            return df[["tecnologia","ano","mes","capacidade"]].rename(columns={"tecnologia":"Tecnologia"})

    # Caso C — constante por tecnologia
    if {"tecnologia","capacidade"}.issubset(cols):
        from datetime import datetime
        ano_atual = datetime.now().year
        base = raw[["tecnologia", "capacidade"]].copy()
        base["capacidade"] = pd.to_numeric(base["capacidade"], errors="coerce").fillna(0.0)
        base["key"] = 1
        meses = pd.DataFrame({"mes": MESES, "key": 1})
        out = (base.merge(meses, on="key", how="left")
                    .drop(columns="key")
                    .assign(ano=ano_atual))[["tecnologia","ano","mes","capacidade"]]
        return out.rename(columns={"tecnologia": "Tecnologia"})

    log.warning("[POH] Formato de capacidade não reconhecido em %s; colunas=%s", path, cols)
    return pd.DataFrame(columns=["Tecnologia", "ano", "mes", "capacidade"])

# ------------------------------------------------------------------------------
# Derivações: cutoff, mapeamento, demanda YTG, cap vs demanda
# ------------------------------------------------------------------------------
def compute_cutoff_for_year(df_current: pd.DataFrame, ano: int) -> int:
    """
    Retorna o último mês (1..12) com volume realizado > 0 no ano dado.
    Se não conseguir inferir, retorna 0.
    """
    if df_current.empty:
        return 0
    df = df_current[df_current.get("ano", 0) == ano].copy()
    if df.empty:
        return 0
    vol = df["volume_uc"] if "volume_uc" in df.columns else pd.to_numeric(df.get("valor", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    mm = df.assign(_vol=vol).query("_vol > 0").get("mes", pd.Series(dtype=int))
    return int(mm.max()) if mm is not None and not mm.empty else 0

def familia_tecnologia_map(df_current: pd.DataFrame, ano: int) -> pd.DataFrame:
    """
    Extrai mapeamento Família → Tecnologia (1:N) a partir do current.
    Colunas de saída: ['Família Comercial','Tecnologia'] (distintas, sem nulos).
    """
    df = df_current.copy()
    # Normalizações de nome
    if "Família Comercial" not in df.columns:
        for c in df.columns:
            if str(c).lower().strip() in {"familia comercial","família comercial","familia","familia_comercial"}:
                df = df.rename(columns={c:"Família Comercial"})
                break
    if "Tecnologia" not in df.columns:
        for c in df.columns:
            if str(c).lower().strip() in {"tecnologia","tec"}:
                df = df.rename(columns={c:"Tecnologia"})
                break
    out = (df[df.get("ano", 0) == ano][["Família Comercial","Tecnologia"]]
             .dropna()
             .drop_duplicates())
    return out

def build_demanda_ytg(df_res: pd.DataFrame, map_fam_tec: pd.DataFrame, ano: int, cutoff: int) -> pd.DataFrame:
    """
    Agrega demanda YTG (vendas) por Tecnologia × Mês a partir do res_working (UI/RES).
    Saída: ['Tecnologia','ano','mes','demanda']
    """
    if df_res.empty:
        return pd.DataFrame(columns=["Tecnologia","ano","mes","demanda"])
    df = df_res.copy()
    # Normaliza Família
    if "Família Comercial" not in df.columns:
        for c in df.columns:
            if str(c).lower().strip() in {"familia comercial","família comercial","familia","familia_comercial"}:
                df = df.rename(columns={c:"Família Comercial"})
                break
    df = df[df.get("ano", 0) == ano]
    if df.empty:
        return pd.DataFrame(columns=["Tecnologia","ano","mes","demanda"])
    df = df[df["mes"] > int(cutoff)]  # YTG
    if df.empty:
        return pd.DataFrame(columns=["Tecnologia","ano","mes","demanda"])
    # Mapeia Família → Tecnologia
    base = df.merge(map_fam_tec, on="Família Comercial", how="left")
    # Fallback: se res_working já trouxer Tecnologia
    if base["Tecnologia"].isna().any() and "Tecnologia" in df_res.columns:
        base.loc[base["Tecnologia"].isna(), "Tecnologia"] = df_res.loc[base["Tecnologia"].isna(), "Tecnologia"]
    base["Tecnologia"] = base["Tecnologia"].fillna("N/A")
    base["volume_uc"] = pd.to_numeric(base["volume_uc"], errors="coerce").fillna(0.0)
    out = (base.groupby(["Tecnologia","ano","mes"], as_index=False)["volume_uc"]
              .sum()
              .rename(columns={"volume_uc":"demanda"}))
    return out

def build_cap_vs_demanda(ano: int,
                         df_current: Optional[pd.DataFrame] = None,
                         df_res_working: Optional[pd.DataFrame] = None,
                         df_capacidade: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, int]:
    """
    Monta DF Tec×Mês com: demanda, capacidade, gap, cobertura, crítico.
    Retorna: (df, cutoff)
    """
    cur = df_current if df_current is not None else load_current()
    res = df_res_working if df_res_working is not None else load_res_working()
    cap = df_capacidade if df_capacidade is not None else load_capacidade_tecnologias()

    cutoff = compute_cutoff_for_year(cur, ano)
    map_fam_tec = familia_tecnologia_map(cur, ano)
    demanda_ytg = build_demanda_ytg(res, map_fam_tec, ano, cutoff)

    # Normaliza capacidade
    if cap.empty:
        cap_long = pd.DataFrame(columns=["Tecnologia","ano","mes","capacidade"])
    else:
        cap_long = cap.copy()
        if "Tecnologia" not in cap_long.columns:
            for c in cap_long.columns:
                if str(c).lower().strip() == "tecnologia":
                    cap_long = cap_long.rename(columns={c:"Tecnologia"})
        for col in ["ano","mes"]:
            if col in cap_long.columns:
                cap_long[col] = pd.to_numeric(cap_long[col], errors="coerce").fillna(0).astype(int)
        if "capacidade" not in cap_long.columns:
            cap_long["capacidade"] = 0.0
        cap_long["capacidade"] = pd.to_numeric(cap_long["capacidade"], errors="coerce").fillna(0.0)
        if "ano" in cap_long.columns:
            cap_long = cap_long[cap_long["ano"] == ano]
        else:
            cap_long["ano"] = ano

    base = demanda_ytg.merge(cap_long[["Tecnologia","ano","mes","capacidade"]], on=["Tecnologia","ano","mes"], how="left")
    base["capacidade"] = pd.to_numeric(base["capacidade"], errors="coerce").fillna(0.0)
    base["gap_capacidade"] = base["capacidade"] - base["demanda"]
    base["cobertura_pct"] = np.where(base["demanda"] > 0, base["capacidade"] / base["demanda"], np.nan)
    base["critico"] = base["capacidade"] < base["demanda"]
    base["mes_nome"] = base["mes"].map(MONTHS_PT)

    log.info("[POH] build_cap_vs_demanda: ano=%s cutoff=%s linhas=%s tec=%s",
             ano, cutoff, base.shape[0], base["Tecnologia"].nunique())

    return base, cutoff

# ------------------------------------------------------------------------------
# Placeholder opcional — simulação de rolagem (não usada pela tela 04_POH)
# ------------------------------------------------------------------------------
def simular_rolagem(ano: int,
                    df_current: Optional[pd.DataFrame] = None,
                    df_res_working: Optional[pd.DataFrame] = None,
                    df_capacidade: Optional[pd.DataFrame] = None,
                    df_estoque_inicial: Optional[pd.DataFrame] = None,
                    df_politica: Optional[pd.DataFrame] = None,
                    df_producao_override: Optional[pd.DataFrame] = None
                    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    **Skeleton** da rolagem:
      - Estoque Final = Inicial + Produção − Vendas
      - Produção padrão (se não houver override): min(capacidade, demanda)
      - Sem política: 'overstock' desativado

    Retorna:
      roll_long: Tec×Fam×Mês com [estoque_inicial, producao, vendas, estoque_final, capacidade, falta, overprod(bool)]
      prod_vs_cap: Tec×Mês com [producao, capacidade, overprod]

    Observação: não é chamada pela 04_POH; fica aqui para futura integração.
    """
    cur = df_current if df_current is not None else load_current()
    res = df_res_working if df_res_working is not None else load_res_working()
    cap = df_capacidade if df_capacidade is not None else load_capacidade_tecnologias()

    cutoff = compute_cutoff_for_year(cur, ano)
    map_fam_tec = familia_tecnologia_map(cur, ano)
    demanda_ytg = build_demanda_ytg(res, map_fam_tec, ano, cutoff)  # Tec×Mês (demanda)

    # Detalhar por família (distribui por mix do RE, se necessário)
    # Aqui, mantemos nivel Tec×Fam×Mês apenas se o res_working já trouxer família mês a mês.
    if "Família Comercial" in res.columns:
        ytg_fam = (res[(res["ano"] == ano) & (res["mes"] > cutoff)]
                   .rename(columns=lambda x: "Família Comercial" if str(x).lower().strip() in {"familia comercial","família comercial","familia","familia_comercial"} else x)
                   .merge(map_fam_tec, on="Família Comercial", how="left"))
        ytg_fam["volume_uc"] = pd.to_numeric(ytg_fam["volume_uc"], errors="coerce").fillna(0.0)
        vendas_fam = (ytg_fam.groupby(["Tecnologia","Família Comercial","ano","mes"], as_index=False)["volume_uc"]
                            .sum().rename(columns={"volume_uc":"vendas"}))
    else:
        # sem família no res_working: colapsa Tec×Mês
        vendas_fam = (demanda_ytg.assign(**{"Família Comercial":"(mix RE)"})
                               .rename(columns={"demanda":"vendas"}))

    # Capacidade Tec×Mês
    cap_long = cap.copy()
    if "Tecnologia" not in cap_long.columns:
        for c in cap_long.columns:
            if str(c).lower().strip() == "tecnologia":
                cap_long = cap_long.rename(columns={c:"Tecnologia"})
    for col in ["ano","mes"]:
        if col in cap_long.columns:
            cap_long[col] = pd.to_numeric(cap_long[col], errors="coerce").fillna(0).astype(int)
    if "capacidade" not in cap_long.columns:
        cap_long["capacidade"] = 0.0
    cap_long["capacidade"] = pd.to_numeric(cap_long["capacidade"], errors="coerce").fillna(0.0)
    if "ano" in cap_long.columns:
        cap_long = cap_long[cap_long["ano"] == ano]
    else:
        cap_long["ano"] = ano

    # Estoque inicial (opcional)
    if df_estoque_inicial is None:
        # tenta ler automaticamente (arquivo mais recente em data/estoque/inicial/)
        path_ini = _latest_matching(ESTOQUE_INICIAL_DIR, "estoque_inicial*.*") if ESTOQUE_INICIAL_DIR.exists() else None
        if path_ini is not None:
            ext = path_ini.suffix.lower()
            if ext in {".xlsx",".xls"}:
                df_estoque_inicial = pd.read_excel(path_ini)
            elif ext in {".csv"}:
                df_estoque_inicial = pd.read_csv(path_ini)
            else:
                df_estoque_inicial = pd.read_excel(path_ini)
        else:
            df_estoque_inicial = pd.DataFrame(columns=["Tecnologia","Família Comercial","ano","mes","estoque_inicial"])

    ini = df_estoque_inicial.copy()
    # normalizações
    if "Tecnologia" not in ini.columns:
        for c in ini.columns:
            if str(c).lower().strip() == "tecnologia":
                ini = ini.rename(columns={c:"Tecnologia"})
    if "Família Comercial" not in ini.columns:
        for c in ini.columns:
            if str(c).lower().strip() in {"familia comercial","família comercial","familia","familia_comercial"}:
                ini = ini.rename(columns={c:"Família Comercial"})
    for col in ["ano","mes","estoque_inicial"]:
        if col in ini.columns:
            ini[col] = pd.to_numeric(ini[col], errors="coerce").fillna(0)
    ini = ini[(ini.get("ano", 0) == ano) & (ini.get("mes", 0) > cutoff)]

    # Produção override (opcional)
    if df_producao_override is None:
        df_producao_override = pd.DataFrame(columns=["Tecnologia","Família Comercial","ano","mes","producao"])
    prod_o = df_producao_override.copy()
    if "Tecnologia" not in prod_o.columns:
        for c in prod_o.columns:
            if str(c).lower().strip() == "tecnologia":
                prod_o = prod_o.rename(columns={c:"Tecnologia"})
    if "Família Comercial" not in prod_o.columns:
        for c in prod_o.columns:
            if str(c).lower().strip() in {"familia comercial","família comercial","familia","familia_comercial"}:
                prod_o = prod_o.rename(columns={c:"Família Comercial"})
    for col in ["ano","mes","producao"]:
        if col in prod_o.columns:
            prod_o[col] = pd.to_numeric(prod_o[col], errors="coerce").fillna(0)
    prod_o = prod_o[(prod_o.get("ano", 0) == ano) & (prod_o.get("mes", 0) > cutoff)]

    # Base Tec×Fam×Mês (YTG)
    base = (vendas_fam.merge(cap_long[["Tecnologia","ano","mes","capacidade"]],
                             on=["Tecnologia","ano","mes"], how="left")
                    .merge(ini, on=["Tecnologia","Família Comercial","ano","mes"], how="left"))
    base["capacidade"] = pd.to_numeric(base["capacidade"], errors="coerce").fillna(0.0)
    base["estoque_inicial"] = pd.to_numeric(base.get("estoque_inicial", 0.0), errors="coerce").fillna(0.0)

    # Produção default: min(capacidade, vendas) se não houver override
    base = base.merge(prod_o, on=["Tecnologia","Família Comercial","ano","mes"], how="left")
    base["producao"] = pd.to_numeric(base["producao"], errors="coerce").fillna(np.nan)
    base["producao"] = np.where(base["producao"].isna(), np.minimum(base["capacidade"], base["vendas"]), base["producao"])

    # Rolagem simples (sem política)
    base["estoque_final"] = base["estoque_inicial"] + base["producao"] - base["vendas"]
    base["falta"] = np.where(base["estoque_final"] < 0, -base["estoque_final"], 0.0)
    base["estoque_final"] = np.where(base["estoque_final"] < 0, 0.0, base["estoque_final"])
    base["overprod"] = np.where(base["producao"] > base["capacidade"], True, False)

    roll_long = base.copy()
    roll_long["mes_nome"] = roll_long["mes"].map(MONTHS_PT)

    # Agregado Tec×Mês
    prod_vs_cap = (roll_long.groupby(["Tecnologia","ano","mes"], as_index=False)
                             .agg(producao=("producao","sum"),
                                  capacidade=("capacidade","sum")))
    prod_vs_cap["overprod"] = np.where(prod_vs_cap["producao"] > prod_vs_cap["capacidade"], True, False)
    prod_vs_cap["mes_nome"] = prod_vs_cap["mes"].map(MONTHS_PT)

    log.info("[POH] simular_rolagem: ano=%s cutoff=%s linhas=%s", ano, cutoff, roll_long.shape[0])
    return roll_long, prod_vs_cap
