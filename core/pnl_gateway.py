# core/pnl_gateway.py
# -*- coding: utf-8 -*-
"""
Gateway central para montar o P&L conforme os controles da UI (sidebar) ou
argumentos vindos do teste_pnl.py.

Compat garantida:
- get_pnl_for_current_settings(year, scenario_label=..., use_sim=..., scale_label=..., **kwargs)
- Se nenhum argumento vier, lê do core.sim (quando existir).
"""

from __future__ import annotations

from pathlib import Path
import logging
from typing import Optional, Tuple

import pandas as pd

import core.models as M
import core.calculator as C
import core.paths as P

# Estado/controles opcionais (UI)
try:
    import core.sim as S
except Exception:
    S = None

_logger = logging.getLogger(__name__)
if not _logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    _logger.addHandler(h)
_logger.setLevel(logging.INFO)


# --------------------------------------------------------------------------------------
# Helpers de estado (com fallback robusto)
# --------------------------------------------------------------------------------------
def _get_scenario_label_from_state() -> str:
    # rótulo de UI; em teste_pnl, normalmente algo como "BP (FY)", "RE (FY)", "Realizado (YTD+YTG)"
    if S and hasattr(S, "get_scenario_label"):
        try:
            val = S.get_scenario_label()
            if val:
                return str(val)
        except Exception:
            pass
    return "Realizado (YTD+YTG)"

def _get_use_simulation_from_state() -> bool:
    if S and hasattr(S, "get_use_simulation"):
        try:
            return bool(S.get_use_simulation())
        except Exception:
            pass
    return False


def _normalize_like_for_parquet(label: str) -> str:
    """
    Não altera os nomes existentes no parquet; apenas traduz rótulos de UI quando for o caso.
    - "BP (FY)" -> "BP"
    - "RE (FY)" -> "RE"
    - outros rótulos: retorna como veio (respeitar parquet)
    """
    if not label:
        return label
    up = label.strip().upper()
    if up.startswith("BP") and "(FY)" in up:
        return "BP"
    if up.startswith("RE") and "(FY)" in up:
        return "RE"
    return label  # já está no formato do parquet ou é "Realizado (YTD+YTG)"


# --------------------------------------------------------------------------------------
# Montagens por cenário
# --------------------------------------------------------------------------------------
def _bp_matrix(year: int, like: str = "BP") -> pd.DataFrame:
    """
    Tenta montar BP para o ano informado.
    1) Primeiro tenta cenario_like=like (ex.: 'BP').
    2) Se vier vazio, lista todos os cenários do CURRENT que começam com 'BP'
       e testa cada um mantendo o NOME EXATO DO PARQUET, escolhendo o que
       tem dados para o ano.
    """
    mat = M.dre_matrix_total(year=year, cenario_like=like)
    have = any(str(c) in mat.columns for c in M.MONTHS_PT)
    if have and not mat.empty:
        return mat

    # fallback: procurar candidatos BP exatos do parquet
    try:
        all_sc = M.list_scenarios_current()
    except Exception:
        all_sc = []

    cand = [s for s in all_sc if str(s).strip().upper().startswith("BP")]
    for scen in sorted(cand):
        mat2 = M.dre_matrix_total(year=year, cenario_like=scen)
        if not mat2.empty and any(str(c) in mat2.columns for c in M.MONTHS_PT):
            return mat2

    # nada encontrado (mantém contrato)
    return mat


def _re_matrix(year: int, like: Optional[str] = None) -> pd.DataFrame:
    """
    Se vier um like explícito (ex.: "RE 05.25" do parquet), usa direto.
    Senão, usa o RE mais recente do ano.
    """
    if like and like.strip().upper() != "RE":
        tag = like  # o chamador passou o texto exato do parquet -> respeitar
    else:
        tag = M.find_latest_re_scenario(year) or "RE"
    _logger.info("Montando RE (FY) para ano=%s, tag=%s", year, tag)
    return M.dre_matrix_total(year=year, cenario_like=tag)

def _realizado_ytd_ytg_matrix(year: int, use_ui: bool) -> pd.DataFrame:
    """
    Realizado (YTD+YTG):
      - YTD: CURRENT (cenario_like='Realizado', exatamente como no parquet).
      - YTG: volumes do RES_WORKING (Grupo 3) OU UI (Grupo 4) se use_ui=True.
      - Linhas copiadas do RE mais recente; RB/Ins/Toll/Frete via base_calculos.xlsx (fallback escala RE).
    """
    # Pivot base do Realizado (YTD)
    piv_real = M.dre_matrix_total(year=year, cenario_like="Realizado")

    # Cutoff (último mês com volume)
    cut_map = M.realized_cutoff_by_year("Realizado")
    cutoff = int(cut_map.get(int(year), 0))

    # CURRENT (long) para o motor
    curr_long = M.load_current_long()

    # Base de cálculos
    base_calc_path: Path = P.BASE_CALCULOS_XLSX

    # RE mais recente (para linhas copiadas)
    re_tag = M.find_latest_re_scenario(year)
    _logger.info("RE detectado para YTG: %s", re_tag)

    # Fonte YTG
    volume_mode = "ui" if use_ui else "res"
    volumes_edit = None
    ui_month_totals = None
    if use_ui and S and hasattr(S, "resolve_monthly_volumes"):
        try:
            volumes_edit, ui_month_totals = S.resolve_monthly_volumes(year)
        except Exception as e:
            _logger.warning("resolve_monthly_volumes falhou; caindo no RES. Erro: %s", e)
            volume_mode = "res"

    # RES_WORKING sempre como baseline
    try:
        volumes_res = M.res_volume_by_family_long()
    except Exception as e:
        _logger.warning("Falha ao ler RES_WORKING: %s", e)
        volumes_res = pd.DataFrame(columns=["Família Comercial", "ano", "mes", "volume"])

    sim = C.build_simulado_pivot(
        df=curr_long,
        piv_real=piv_real,
        year=int(year),
        cutoff=int(cutoff),
        base_calc_path=base_calc_path,
        volumes_edit=volumes_edit,       # UI (opcional)
        volumes_res=volumes_res,         # RES_WORKING baseline
        volume_mode=volume_mode,         # 'ui' quando use_ui=True; senão 'res'
        dme_pct=None,                    # usa DME_TAX do models
        ui_month_totals=ui_month_totals, # totais UI (opcional)
        conv_source="excel",
    )
    return sim


# --------------------------------------------------------------------------------------
# API pública (compatível com teste_pnl.py)
# --------------------------------------------------------------------------------------
def get_pnl_for_current_settings(
    year: int,
    scenario_label: Optional[str] = None,
    use_sim: Optional[bool] = None,
    scale_label: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Mantém compat: aceita kwargs extras sem quebrar.
    Regras:
      - Se scenario_label vier do chamador, respeitamos (e só "traduzimos" rótulos de UI como BP/RE (FY)).
      - Se não vier, lemos do core.sim (quando existir).
      - Para 'Realizado' (qualquer variação contendo 'realiz'), montamos YTD+YTG (YTG do RES_WORKING por padrão);
        se use_sim=True, YTG vem da UI (Grupo 4).
    """
    scen_label_in = scenario_label if scenario_label is not None else _get_scenario_label_from_state()
    scen_like = _normalize_like_for_parquet(scen_label_in)

    use_ui = use_sim if use_sim is not None else _get_use_simulation_from_state()

    _logger.info(
        "get_pnl_for_current_settings: year=%s, scenario_label_in=%s, scen_like=%s, use_ui=%s",
        year, scen_label_in, scen_like, use_ui
    )

    up = (scen_like or "").upper()

    # Caso UI passe "Realizado (YTD+YTG)" ou qualquer coisa contendo 'realiz'
    if "REALIZ" in up:
        return _realizado_ytd_ytg_matrix(year, use_ui=use_ui)

    # Caso UI passe "BP (FY)" -> traduzimos para 'BP'; se vier um nome exato do parquet, respeitar
    if up.startswith("BP"):
        like = _normalize_like_for_parquet(scen_like)  # 'BP (FY)' -> 'BP', ou nome exato do parquet
        return _bp_matrix(year, like=like)

    # Caso UI passe "RE (FY)" -> usamos RE mais recente; se vier "RE 05.25" do parquet, respeitar
    if up.startswith("RE"):
        like = _normalize_like_for_parquet(scen_like)
        return _re_matrix(year, like=like)

    # Fallback: tentar exatamente o que o chamador pediu (útil quando há nomes específicos no parquet)
    _logger.info("Cenário não mapeado explicitamente; tentando like='%s'", scen_like)
    return M.dre_matrix_total(year=year, cenario_like=scen_like)
