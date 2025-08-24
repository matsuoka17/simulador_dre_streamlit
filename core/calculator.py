# core/calculator.py
# -*- coding: utf-8 -*-
"""
Motor de cálculo do P&L (Simulado) – determinístico e puro.

- Mantém YTD = Realizado (cópia do pivot base).
- Calcula YTG do zero (mês a mês) a partir dos volumes-alvo (UI → RES → RE).
- Linhas copiadas do RE (sem escala): EI Var/Fixa, T1, T2, Perdas, Opex, Chargeback, Depreciação.
- Demais linhas (RB, Insumos, Toll Packer, Fretes, composições) são recalculadas em função do volume.

Entradas principais:
- df (LONG): current.parquet (governança via core.models), contendo 'indicador_id','valor','cenario','ano','mes', etc.
- piv_real: pivot do cenário Realizado (linhas P&L × meses Jan..Dez + 'Total Ano').
- volumes_edit/res: volumes por família (long) opcionais; se vier apenas total mensal, distribuir por mix do RE.
- base_calculos.xlsx: parâmetros R$/UC por família e mês (Preço, Insumo, Toll, Frete). Se faltar, fallback escala pelo RE.

Saída:
- DataFrame no mesmo formato de piv_real (linhas idênticas / colunas 'Jan'..'Dez' + 'Total Ano'), inteiros arredondados.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
from core.models import CONV_TAX, DME_TAX  # <- Import tax dme e impostos

import logging
import numpy as np
import pandas as pd

try:
    import streamlit as st
except Exception:
    st = None

# -----------------------------------------------------------------------------
# Constantes de negócio
# -----------------------------------------------------------------------------
# REMOVIDO: definições duplicadas - usar do models.py
VOLUME_ID = "volume_uc"

MONTHS_PT = {
    1: "Jan",  2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
    7: "Jul",  8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"
}
MONTH_ORDER = [MONTHS_PT[i] for i in range(1, 13)]

# Linhas do P&L (ids canônicos, precisam casar com o pivot/visão)
PNL_ROWS = [
    "volume_uc",
    "receita_bruta",
    "convenio_impostos",
    "receita_liquida",
    "insumos",
    "custos_toll_packer",
    "fretes_t1",
    "estrutura_industrial_variavel",
    "custos_variaveis",
    "margem_variavel",
    "estrutura_industrial_fixa",
    "margem_bruta",
    "custos_log_t1_reg",
    "despesas_secundarias_t2",
    "perdas",
    "margem_contribuicao",
    "dme",
    "margem_contribuicao_liquida",
    "opex_leao_reg",
    "chargeback",
    "resultado_operacional",
    "depreciacao",
    "ebitda",
]

DISPLAY_LABELS = {
    "volume_uc": "Volume (UC)",
    "receita_bruta": "Receita Bruta",
    "convenio_impostos": "Convênio / Impostos",
    "receita_liquida": "Receita Líquida",
    "insumos": "Insumos",
    "custos_toll_packer": "Custos Toll Packer",
    "fretes_t1": "Fretes T1",
    "estrutura_industrial_variavel": "Estrutura Industrial Variável",
    "custos_variaveis": "Custos Variáveis",
    "margem_variavel": "Margem Variável",
    "estrutura_industrial_fixa": "Estrutura Industrial Fixa",
    "margem_bruta": "Margem Bruta",
    "custos_log_t1_reg": "Custos Log T1 Reg",
    "despesas_secundarias_t2": "Despesas Secundárias T2",
    "perdas": "Perdas",
    "margem_contribuicao": "Margem de Contribuição",
    "dme": "DME",
    "margem_contribuicao_liquida": "Margem de Contribuição Líquida",
    "opex_leao_reg": "Opex Leão Reg",
    "chargeback": "Charge Back",
    "resultado_operacional": "Resultado Operacional",
    "depreciacao": "Depreciação",
    "ebitda": "EBITDA",
}

# Linhas copiadas do RE (sem escala) para YTG:
RE_COPY_ROWS = {
    "estrutura_industrial_variavel",
    "estrutura_industrial_fixa",
    "custos_log_t1_reg",
    "despesas_secundarias_t2",
    "perdas",
    "opex_leao_reg",
    "chargeback",
    "depreciacao",
}

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------
def _file_signature(p: Path) -> str:
    p = Path(p)
    try:
        stt = p.stat()
        return f"{p.as_posix()}|{stt.st_mtime_ns}|{stt.st_size}"
    except FileNotFoundError:
        return f"{p.as_posix()}|MISSING"

def _fam_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("Família Comercial", "Familia Comercial", "familia_comercial", "Familia"):
        if c in df.columns:
            return c
    return None

def _safe_int(v) -> int:
    try:
        return int(np.rint(float(v)))
    except Exception:
        return 0

def _month_str(m: int) -> str:
    return MONTHS_PT.get(int(m), str(m))

def _norm_cenario(s: str) -> str:
    return (s or "").strip().upper()

def _is_realizado(s: str) -> bool:
    s2 = _norm_cenario(s)
    return "REALIZ" in s2

def _is_re(s: str) -> bool:
    s2 = _norm_cenario(s)
    return s2.startswith("RE") and not _is_realizado(s2)

def _parse_re_tag(s: str) -> Tuple[int, int]:
    """
    Tenta ler 'RE MM.AA' (ou variações 'REMM.AA', 'RE MM AA', etc.)
    Retorna (yy, mm) ou (-1,-1) se não conseguir.
    """
    s2 = _norm_cenario(s)
    if not s2.startswith("RE"):
        return (-1, -1)
    # Extrai números na ordem mm, aa (2 dígitos)
    import re as _re
    m = _re.search(r"RE\D*?(\d{2})\D*?(\d{2})", s2)
    if not m:
        return (-1, -1)
    mm = int(m.group(1))
    yy = int(m.group(2))
    if 1 <= mm <= 12:
        return (yy, mm)
    return (-1, -1)

def _latest_re_scenario(df: pd.DataFrame, year: int) -> Optional[str]:
    """
    Escolhe o último RE disponível. Preferência para o RE cujo 'AA' == year % 100.
    Caso não exista para o ano, pega o maior (yy,mm) global.
    """
    cand = df["cenario"].dropna().astype(str).unique().tolist()
    re_tags = [c for c in cand if _is_re(c)]
    if not re_tags:
        return None
    yy_target = int(year) % 100

    # Filtra por yy alvo
    same_year = [(c, *_parse_re_tag(c)) for c in re_tags]
    same_year = [(c, yy, mm) for (c, yy, mm) in same_year if yy == yy_target]
    if same_year:
        best = max(same_year, key=lambda x: x[2])  # maior mm
        return best[0]

    # Fallback: maior (yy,mm) global
    tags = [(c, *_parse_re_tag(c)) for c in re_tags]
    tags = [(c, yy, mm) for (c, yy, mm) in tags if yy >= 0 and mm >= 0]
    if not tags:
        return re_tags[-1]
    best = max(tags, key=lambda x: (x[1], x[2]))  # maior yy, depois mm
    return best[0]

def _get_series_from_scenario(df: pd.DataFrame, year: int, scenario: Optional[str], indic_id: str) -> Dict[int, float]:
    if scenario is None:
        return {m: 0.0 for m in range(1, 13)}
    sub = df[(df["ano"] == year) & (df["cenario"] == scenario) & (df["indicador_id"] == indic_id)]
    if sub.empty:
        return {m: 0.0 for m in range(1, 13)}
    g = sub.groupby("mes")["valor"].sum()
    return {int(m): float(g.get(m, 0.0)) for m in range(1, 13)}

def _get_series_by_family(df: pd.DataFrame, year: int, scenario: Optional[str], indic_id: str) -> Dict[Tuple[str, int], float]:
    if scenario is None:
        return {}
    famc = _fam_col(df)
    if famc is None:
        return {}
    sub = df[(df["ano"] == year) & (df["cenario"] == scenario) & (df["indicador_id"] == indic_id)]
    if sub.empty:
        return {}
    g = sub.groupby([famc, "mes"])["valor"].sum()
    return {(str(f), int(m)): float(v) for (f, m), v in g.items()}

# -----------------------------------------------------------------------------
# Parâmetros (planilha Excel)
# -----------------------------------------------------------------------------
def _load_base_calculos(path: Path) -> pd.DataFrame:
    """
    Espera colunas (variações aceitas):
      - Indicador
      - Família Comercial
      - Mês (Jan..Dez)
      - Valor (R$/UC)   (Preço, Insumo, Toll, Frete, eventualmente EI var/fic se existir)
    """
    df = pd.read_excel(path)
    rename = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl.startswith("indic"):
            rename[c] = "Indicador"
        elif cl.startswith("fam"):
            rename[c] = "Família Comercial"
        elif cl in {"mes", "mês", "meses"}:
            rename[c] = "Mês"
        elif cl in {"valor", "val", "r$/uc", "rs/uc"}:
            rename[c] = "Valor"
        elif cl in {"%", "pct", "percentual", "percent"}:
            rename[c] = "%"
    if rename:
        df = df.rename(columns=rename)

    needed = {"Indicador", "Família Comercial", "Mês"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"base_calculos.xlsx: colunas ausentes: {missing}")

    # Normaliza mês em 'Jan'..'Dez'
    mfix = {str(v).lower(): k for k, v in MONTHS_PT.items()}
    df["Mês"] = df["Mês"].astype(str).str.strip()
    df["Mês"] = df["Mês"].apply(lambda s: MONTHS_PT.get(mfix.get(s.lower(), -1), s))
    return df

def _param_lookup(params: pd.DataFrame, familia: str, mes_num: int, indicador: str, field: str = "Valor") -> float:
    mes = _month_str(mes_num)
    sub = params[
        (params["Família Comercial"] == familia)
        & (params["Mês"] == mes)
        & (params["Indicador"] == indicador)
    ]
    if sub.empty or field not in sub.columns:
        return 0.0
    v = sub.iloc[0][field]
    try:
        return float(v)
    except Exception:
        return 0.0

# Cache para Excel (por assinatura)
if st is not None:
    @st.cache_data(show_spinner=False)
    def _load_base_calculos_cached(sig: str, path: str) -> pd.DataFrame:
        return _load_base_calculos(Path(path))
else:
    def _load_base_calculos_cached(sig: str, path: str) -> pd.DataFrame:
        return _load_base_calculos(Path(path))


# -----------------------------------------------------------------------------
# Núcleo – construir YTG e aplicar no pivot
# -----------------------------------------------------------------------------
def build_simulado_pivot(
    df: pd.DataFrame,
    piv_real: pd.DataFrame,
    year: int,
    cutoff: int,
    base_calc_path: Path,
    volumes_edit: Optional[pd.DataFrame] = None,
    volumes_res: Optional[pd.DataFrame] = None,
    volume_mode: str = "re",
    dme_pct: Optional[float] = None,          # <- agora opcional (None = usar DME_TAX oficial)
    ui_month_totals: Optional[Dict[int, int]] = None,
    conv_source: str = "excel",               # mantido para compat; desconsiderado no cálculo
) -> pd.DataFrame:
    """
    Retorna um pivot no MESMO formato de `piv_real`, com:
      - Meses <= cutoff: mantidos como no realizado.
      - Meses  > cutoff: recalculados por regras:
          * RB/Ins/Toll/Frete via base_calculos.xlsx (R$/UC × volume).
          * EI Var/Fixa + T1/T2/Perdas/Opex/CB/Depre copiados do último RE.
          * Convênio e DME via alíquotas fixas (-26,5% e -10,2%).
          * Demais composições derivadas.
    """

    # 0) Logs/checagens básicas
    if base_calc_path is None:
        raise ValueError("base_calc_path não informado.")
    if piv_real is None or piv_real.empty:
        raise ValueError("piv_real vazio – gere pivot do Realizado antes de chamar o cálculo.")

    # 1) Carrega parâmetros (Excel) com cache por assinatura
    sig = _file_signature(base_calc_path)
    params = _load_base_calculos_cached(sig, str(base_calc_path))

    # 2) Descobre o último RE do ano (ou global, se não houver para o ano)
    re_scen = _latest_re_scenario(df, year)
    _logger.info(f"[calculator] Último RE detectado (ano {year}): {re_scen}")

    # 3) Mapas do RE – por família (para mix e fallback de custo) e por mês (para EI/linhas copiadas)
    vol_re_fam = _get_series_by_family(df, year, re_scen, VOLUME_ID)
    re_total_mes = {m: 0.0 for m in range(1, 13)}
    for (_, mm), v in vol_re_fam.items():
        re_total_mes[mm] = re_total_mes.get(mm, 0.0) + float(v)

    re_rb_fam   = _get_series_by_family(df, year, re_scen, "receita_bruta")
    re_ins_fam  = _get_series_by_family(df, year, re_scen, "insumos")
    re_toll_fam = _get_series_by_family(df, year, re_scen, "custos_toll_packer")
    re_frt_fam  = _get_series_by_family(df, year, re_scen, "fretes_t1")

    # Linhas copiadas do RE (totais mensais)
    re_eivar_m = _get_series_from_scenario(df, year, re_scen, "estrutura_industrial_variavel")
    re_eifxa_m = _get_series_from_scenario(df, year, re_scen, "estrutura_industrial_fixa")
    re_t1_m    = _get_series_from_scenario(df, year, re_scen, "custos_log_t1_reg")
    re_t2_m    = _get_series_from_scenario(df, year, re_scen, "despesas_secundarias_t2")
    re_perdas_m= _get_series_from_scenario(df, year, re_scen, "perdas")
    re_opex_m  = _get_series_from_scenario(df, year, re_scen, "opex_leao_reg")
    re_cb_m    = _get_series_from_scenario(df, year, re_scen, "chargeback")
    re_dep_m   = _get_series_from_scenario(df, year, re_scen, "depreciacao")

    # 4) Volumes UI/RES – por família (se houver) e totais mensais auxiliares
    def _to_fam_map(vol_df: Optional[pd.DataFrame]) -> Dict[Tuple[str, int], float]:
        if vol_df is None or vol_df.empty:
            return {}
        famc = _fam_col(vol_df)
        if famc is None:
            return {}
        req = {"ano", "mes"}
        if not req.issubset(set(vol_df.columns)):
            return {}
        sub = vol_df[vol_df["ano"] == year]
        if "volume" not in sub.columns:
            # tenta detectar coluna numérica com volume
            for c in sub.columns:
                if str(c).lower() in {"volume", "volume_uc", "qtd", "quantidade", "uc", "valor"}:
                    sub = sub.rename(columns={c: "volume"})
                    break
        if "volume" not in sub.columns:
            return {}
        g = sub.groupby([famc, "mes"])["volume"].sum()
        return {(str(f), int(m)): float(v) for (f, m), v in g.items()}

    vol_ui_fam  = _to_fam_map(volumes_edit)
    vol_res_fam = _to_fam_map(volumes_res)

    res_month_totals: Optional[Dict[int, float]] = None
    if volumes_res is not None and {"ano", "mes"}.issubset(volumes_res.columns):
        sr = volumes_res[volumes_res["ano"] == year]
        col_v = "volume"
        if "volume" not in sr.columns:
            for c in sr.columns:
                if str(c).lower() in {"volume", "volume_uc", "qtd", "quantidade", "uc", "valor"}:
                    col_v = c
                    break
        if col_v in sr.columns:
            g = sr.groupby("mes")[col_v].sum()
            res_month_totals = {int(m): float(g.get(m, 0.0)) for m in range(1, 13)}

    if ui_month_totals is None and volumes_edit is not None and {"ano", "mes"}.issubset(volumes_edit.columns):
        se = volumes_edit[volumes_edit["ano"] == year]
        if "volume" in se.columns:
            g = se.groupby("mes")["volume"].sum()
            ui_month_totals = {int(m): float(g.get(m, 0.0)) for m in range(1, 13)}

    # Mix YTD (RE) – para distribuir total quando faltar família/mês do RE
    re_ytd_family_totals: Dict[str, float] = {}
    for (fam, mm), v in vol_re_fam.items():
        if cutoff == 0 or int(mm) <= int(cutoff):
            re_ytd_family_totals[fam] = re_ytd_family_totals.get(fam, 0.0) + float(v)
    re_ytd_total = float(sum(re_ytd_family_totals.values()))

    # Fonte primária conforme volume_mode
    def familias_do_mes(m: int) -> set[str]:
        if volume_mode == "ui":
            fams_ui = {f for (f, mm) in vol_ui_fam.keys() if mm == m}
            if fams_ui:
                return fams_ui
            fams_re_m = {f for (f, mm) in vol_re_fam.keys() if mm == m}
            return fams_re_m if fams_re_m else set(re_ytd_family_totals.keys())

        if volume_mode == "res":
            fams_res = {f for (f, mm) in vol_res_fam.keys() if mm == m}
            if fams_res:
                return fams_res
            fams_re_m = {f for (f, mm) in vol_re_fam.keys() if mm == m}
            return fams_re_m if fams_re_m else set(re_ytd_family_totals.keys())

        # "re"
        fams_re_m = {f for (f, mm) in vol_re_fam.keys() if mm == m}
        return fams_re_m if fams_re_m else set(re_ytd_family_totals.keys())

    def vol_for(fam: str, m: int) -> float:
        # RES como fonte
        if volume_mode == "res":
            if (fam, m) in vol_res_fam:
                return float(vol_res_fam[(fam, m)])
            if res_month_totals and res_month_totals.get(m, 0.0) > 0:
                total_res = float(res_month_totals[m])
                base_total = float(re_total_mes.get(m, 0.0))
                if base_total > 0:
                    share = float(vol_re_fam.get((fam, m), 0.0)) / base_total
                elif re_ytd_total > 0:
                    share = float(re_ytd_family_totals.get(fam, 0.0)) / re_ytd_total
                else:
                    share = 0.0
                return total_res * share
            return 0.0

        # UI como fonte
        if volume_mode == "ui":
            if (fam, m) in vol_ui_fam:
                return float(vol_ui_fam[(fam, m)])
            if ui_month_totals and ui_month_totals.get(m, 0.0) > 0:
                total_ui = float(ui_month_totals[m])
                base_total = float(re_total_mes.get(m, 0.0))
                if base_total > 0:
                    share = float(vol_re_fam.get((fam, m), 0.0)) / base_total
                elif re_ytd_total > 0:
                    share = float(re_ytd_family_totals.get(fam, 0.0)) / re_ytd_total
                else:
                    share = 0.0
                return total_ui * share
            # Fallback: RE por família
            return float(vol_re_fam.get((fam, m), 0.0))

        # RE como fonte
        return float(vol_re_fam.get((fam, m), 0.0))

    # 5) Constrói valores YTG (mês a mês)
    # Inicializa dicionário para todos os ids
    ytg_vals: Dict[str, Dict[str, float]] = {rid: {} for rid in PNL_ROWS}

    for m in range(cutoff + 1, 13):
        ms = MONTHS_PT[m]

        # ----- Volumes (UC) -----
        vol_total_mes = 0.0
        fams = familias_do_mes(m)
        for fam in fams:
            vol_total_mes += float(vol_for(fam, m))
        ytg_vals["volume_uc"][ms] = vol_total_mes

        # ----- Bases por família (R$/UC × volume) com fallback de escala pelo RE -----
        rb_total = 0.0
        ins_total = 0.0
        toll_total = 0.0
        frete_total = 0.0

        for fam in fams:
            v_new = float(vol_for(fam, m))
            v_re  = float(vol_re_fam.get((fam, m), 0.0))
            ratio = (v_new / v_re) if v_re > 0 else 0.0

            # Receita Bruta
            preco_uc = _param_lookup(params, fam, m, "Receita Bruta", "Valor")
            if preco_uc != 0.0:
                rb_total += v_new * float(preco_uc)
            else:
                rb_total += float(re_rb_fam.get((fam, m), 0.0)) * ratio

            # Insumos (negativo)
            ins_uc = _param_lookup(params, fam, m, "Insumos", "Valor")
            if ins_uc != 0.0:
                ins_total += -(abs(ins_uc) * v_new)
            else:
                ins_total += float(re_ins_fam.get((fam, m), 0.0)) * ratio

            # Toll Packer (negativo)
            toll_uc = _param_lookup(params, fam, m, "Custos Toll Packer", "Valor")
            if toll_uc != 0.0:
                toll_total += -(abs(toll_uc) * v_new)
            else:
                toll_total += float(re_toll_fam.get((fam, m), 0.0)) * ratio

            # Fretes T1 (negativo)
            frete_uc = _param_lookup(params, fam, m, "Fretes T1", "Valor")
            if frete_uc != 0.0:
                frete_total += -(abs(frete_uc) * v_new)
            else:
                frete_total += float(re_frt_fam.get((fam, m), 0.0)) * ratio

        ytg_vals["receita_bruta"][ms] = rb_total
        ytg_vals["insumos"][ms] = ins_total
        ytg_vals["custos_toll_packer"][ms] = toll_total
        ytg_vals["fretes_t1"][ms] = frete_total

        # ----- Copiados do RE (sem escalar) -----
        ytg_vals["estrutura_industrial_variavel"][ms] = float(re_eivar_m.get(m, 0.0))
        ytg_vals["estrutura_industrial_fixa"][ms]     = float(re_eifxa_m.get(m, 0.0))
        ytg_vals["custos_log_t1_reg"][ms]             = float(re_t1_m.get(m, 0.0))
        ytg_vals["despesas_secundarias_t2"][ms]       = float(re_t2_m.get(m, 0.0))
        ytg_vals["perdas"][ms]                        = float(re_perdas_m.get(m, 0.0))
        ytg_vals["opex_leao_reg"][ms]                 = float(re_opex_m.get(m, 0.0))
        ytg_vals["chargeback"][ms]                    = float(re_cb_m.get(m, 0.0))
        ytg_vals["depreciacao"][ms]                   = float(re_dep_m.get(m, 0.0))

        # ----- Composições -----
        rb = float(ytg_vals["receita_bruta"][ms])

        conv_rate = float(CONV_TAX)
        conv = conv_rate * rb
        ytg_vals["convenio_impostos"][ms] = conv

        rl = rb + conv
        ytg_vals["receita_liquida"][ms] = rl

        cv = (
            float(ytg_vals["insumos"][ms]) +
            float(ytg_vals["custos_toll_packer"][ms]) +
            float(ytg_vals["fretes_t1"][ms]) +
            float(ytg_vals["estrutura_industrial_variavel"][ms])
        )
        ytg_vals["custos_variaveis"][ms] = cv

        mv = rl + cv                                    # CV já negativo
        ytg_vals["margem_variavel"][ms] = mv

        mb = mv + float(ytg_vals["estrutura_industrial_fixa"][ms])  # EIF deve vir negativa do RE
        ytg_vals["margem_bruta"][ms] = mb

        t1 = float(ytg_vals["custos_log_t1_reg"][ms])
        t2 = float(ytg_vals["despesas_secundarias_t2"][ms])
        ps = float(ytg_vals["perdas"][ms])
        mc = mb + (t1 + t2 + ps)                        # todos negativos (RE)
        ytg_vals["margem_contribuicao"][ms] = mc

        if dme_pct is None:
            dme_rate = float(DME_TAX)
        else:
            # garante sinal negativo se alguém passar 0.102 por hábito
            dme_rate = -abs(float(dme_pct))
        dme_val = dme_rate * rl
        ytg_vals["dme"][ms] = dme_val

        mcl = mc + dme_val
        ytg_vals["margem_contribuicao_liquida"][ms] = mcl

        opex = float(ytg_vals["opex_leao_reg"][ms])
        cb   = float(ytg_vals["chargeback"][ms])
        ro   = mcl + opex + cb
        ytg_vals["resultado_operacional"][ms] = ro

        dep  = float(ytg_vals["depreciacao"][ms])
        ebt  = ro + dep
        ytg_vals["ebitda"][ms] = ebt

    # 6) Aplicar YTG no pivot de saída
    out = piv_real.copy()
    for col in MONTH_ORDER:
        if col not in out.columns:
            out[col] = 0

    # Garante que todas as linhas existam (se o pivot não tiver alguma)
    have_ids = set(out["Conta" if "Conta" in out.columns else "indicador_id"].astype(str).tolist())
    missing = [rid for rid in PNL_ROWS if rid not in have_ids]
    if missing:
        for rid in missing:
            row = {"Conta": DISPLAY_LABELS.get(rid, rid)}
            if "indicador_id" in out.columns:
                row["indicador_id"] = rid
            for mc in MONTH_ORDER:
                row[mc] = 0
            row["Total Ano"] = 0
            out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)

    # Aplica apenas meses > cutoff
    id_col = "Conta" if "Conta" in out.columns else "indicador_id"
    for rid, month_map in ytg_vals.items():
        for ms, val in month_map.items():
            mask = out[id_col] == rid
            if not mask.any():
                # Se linha não existe, criar
                row = {id_col: rid}
                for mc in MONTH_ORDER:
                    row[mc] = 0 if mc != ms else _safe_int(val)
                row["Total Ano"] = _safe_int(val) if ms else 0
                out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)
            else:
                out.loc[mask, ms] = _safe_int(val)

    # Recalcula totals
    out["Total Ano"] = out[MONTH_ORDER].sum(axis=1).astype(int)

    # Ordena por PNL_ROWS (se possível) mantendo linhas extras no final
    out["__ord__"] = pd.Categorical(out[id_col], categories=PNL_ROWS, ordered=True)
    out = out.sort_values("__ord__", na_position="last").drop(columns="__ord__").reset_index(drop=True)

    _logger.info("[calculator] Simulação concluída: YTG aplicado e totais recalculados.")
    return out