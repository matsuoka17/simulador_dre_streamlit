# core/sim.py
from __future__ import annotations
from pathlib import Path
import hashlib, json
import pandas as pd
import streamlit as st
from typing import Dict, Tuple

# ==== chaves globais (session_state) ====
_SCEN_LABEL_KEY = "scenario_label_global"  # "BP (FY)" | "RE (FY)" | "Realizado (YTD+YTG)"
_USE_SIM_KEY = "use_sim_global"  # bool (só tem efeito no Realizado)
_SCALE_LABEL_KEY = "scale_label_global"  # "1x" | "1.000x" | "1.000.000x"

_SCEN_CHOICES = ["BP (FY)", "RE (FY)", "Realizado (YTD+YTG)"]
_SCALE_CHOICES = ["1x", "1.000x", "1.000.000x"]
_SCALE_MAP = {"1x": 1, "1.000x": 1_000, "1.000.000x": 1_000_000}


def ensure_global_defaults(
        *,
        default_scenario: str = "Realizado (YTD+YTG)",
        default_use_sim: bool = True,
        default_scale: str = "1x",
) -> None:
    """Chame ANTES de renderizar widgets (ex.: no init da página ou na sidebar)."""
    st.session_state.setdefault(_SCEN_LABEL_KEY, default_scenario)
    st.session_state.setdefault(_USE_SIM_KEY, default_use_sim)
    st.session_state.setdefault(_SCALE_LABEL_KEY, default_scale)


# ----------------- escala (somente formatação) -----------------
def render_global_scale_selector(sidebar: bool = True, label: str = "Escala") -> None:
    ensure_global_defaults()
    widget = st.sidebar if sidebar else st
    idx = _SCALE_CHOICES.index(st.session_state.get(_SCALE_LABEL_KEY, _SCALE_CHOICES[0]))
    widget.selectbox(label, _SCALE_CHOICES, index=idx, key=_SCALE_LABEL_KEY)


def get_global_scale() -> tuple[str, int]:
    """Retorna (rótulo, fator)."""
    lab = st.session_state.get(_SCALE_LABEL_KEY, _SCALE_CHOICES[0])
    return lab, _SCALE_MAP[lab]


# retrocompatibilidade (páginas antigas)
def get_global_scale_factor() -> int:
    """Apenas o fator numérico."""
    return get_global_scale()[1]


# ----------------- cenário + toggle projeção -----------------
def render_global_scenario_selector(sidebar: bool = True, label: str = "Cenário") -> None:
    ensure_global_defaults()
    widget = st.sidebar if sidebar else st
    cur = st.session_state.get(_SCEN_LABEL_KEY, _SCEN_CHOICES[-1])
    idx = _SCEN_CHOICES.index(cur) if cur in _SCEN_CHOICES else len(_SCEN_CHOICES) - 1
    widget.selectbox(label, _SCEN_CHOICES, index=idx, key=_SCEN_LABEL_KEY)


def render_global_sim_toggle(label: str = "Usar Projeção (volumes YTG)", sidebar: bool = True) -> None:
    """Só aparece/vale quando cenário = Realizado (YTD+YTG)."""
    ensure_global_defaults()
    widget = st.sidebar if sidebar else st
    scen = st.session_state.get(_SCEN_LABEL_KEY, _SCEN_CHOICES[-1])
    if scen != "Realizado (YTD+YTG)":
        st.session_state[_USE_SIM_KEY] = False
        return
    widget.checkbox(label, key=_USE_SIM_KEY,
                    help="ON: YTG usa volumes da UI (fallback RES). OFF: usa apenas RES/parquet.")


def get_scenario_label() -> str:
    return st.session_state.get(_SCEN_LABEL_KEY, _SCEN_CHOICES[-1])


def get_use_sim_effective() -> bool:
    """True só se cenário = Realizado e toggle ligado."""
    return (get_scenario_label() == "Realizado (YTD+YTG)") and bool(st.session_state.get(_USE_SIM_KEY, True))


# alias com o nome usado no 03_P&L.py
def get_global_use_simulation() -> bool:
    return get_use_sim_effective()


# ----------------- volumes (fonte única) -----------------
def resolve_monthly_volumes(year: int) -> dict[int, int] | None:
    """
    Fonte única e determinística para os volumes YTG:
    - Se get_global_use_simulation() == True: usar UI (volumes_wide/volumes_edit); fallback RES.
    - Senão: usar RES somente (parquet).
    Retorno {mes:int -> total:int} ou None.
    """
    use_sim = get_global_use_simulation()

    # 1) tentar UI (se habilitado)
    if use_sim:
        vw = st.session_state.get("volumes_wide")
        if isinstance(vw, dict) and year in vw and isinstance(vw[year], pd.DataFrame):
            dfw = vw[year]
            months_lower = {"jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6, "jul": 7, "ago": 8, "set": 9,
                            "out": 10, "nov": 11, "dez": 12}
            col_map = {}
            for c in dfw.columns:
                lc = str(c).strip().lower()[:3]
                if lc in months_lower:
                    col_map[months_lower[lc]] = c
            got = {m: int(pd.to_numeric(dfw[c], errors="coerce").fillna(0).sum()) for m, c in col_map.items()}
            if got:
                return {m: got.get(m, 0) for m in range(1, 13)}

        ve = st.session_state.get("volumes_edit")
        if isinstance(ve, pd.DataFrame) and {"ano", "mes", "volume"}.issubset(ve.columns):
            sub = ve[ve["ano"] == year]
            if not sub.empty:
                g = pd.to_numeric(sub["volume"], errors="coerce").fillna(0).groupby(sub["mes"]).sum()
                return {m: int(g.get(m, 0)) for m in range(1, 13)}

    # 2) RES (parquet)
    try:
        from core.models import res_volume_total_by_month
        res = res_volume_total_by_month(year)
        if res:
            return {m: int(res.get(m, 0)) for m in range(1, 13)}
    except Exception:
        pass

    return None


def resolve_monthly_volumes_by_family(year: int) -> Dict[Tuple[str, int], float]:
    """
    Retorna volumes por família e mês: {(família, mês): volume}
    Necessário para o calculator funcionar corretamente.
    """
    use_sim = get_global_use_simulation()

    # 1) Tentar UI primeiro (se habilitado)
    if use_sim:
        ve = st.session_state.get("volumes_edit")
        if isinstance(ve, pd.DataFrame):
            cols_needed = {"ano", "mes", "volume"}
            fam_col = None
            for c in ve.columns:
                if str(c).lower().startswith("fam"):
                    fam_col = c
                    break

            if cols_needed.issubset(ve.columns) and fam_col:
                sub = ve[ve["ano"] == year]
                if not sub.empty:
                    g = sub.groupby([fam_col, "mes"])["volume"].sum()
                    return {(str(f), int(m)): float(v) for (f, m), v in g.items()}

    # 2) Fallback para RES/RE
    try:
        from core.models import load_res_raw
        df = load_res_raw()
        if not df.empty and "ano" in df.columns:
            # Detectar colunas
            fam_col = None
            vol_col = None
            for c in df.columns:
                if str(c).lower().startswith("fam"):
                    fam_col = c
                elif str(c).lower() in {"volume", "volume_uc", "qtd", "uc"}:
                    vol_col = c

            if fam_col and vol_col:
                sub = df[df["ano"] == year]
                g = sub.groupby([fam_col, "mes"])[vol_col].sum()
                return {(str(f), int(m)): float(v) for (f, m), v in g.items()}
    except Exception:
        pass

    return {}


# ----------------- token de cache para o P&L híbrido -----------------
def _path_mtime_safe(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except Exception:
        return 0.0


def _hash_df_in_session(key: str) -> str:
    obj = st.session_state.get(key)
    try:
        if isinstance(obj, pd.DataFrame):
            return hashlib.md5(pd.util.hash_pandas_object(obj, index=True).values).hexdigest()
        if isinstance(obj, dict):
            return hashlib.md5(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()
    except Exception:
        pass
    return "NA"


def build_cache_token(year: int) -> str:
    """Inclui: ano, cenário, toggle, hash dos volumes UI, mtimes dos parquets e xlsx, e uma versão."""
    from core.paths import PARQUET_DIR, PREMISSAS_PNL_DIR
    pieces = [
        f"y={year}",
        f"scen={get_scenario_label()}",
        f"use_sim={int(get_global_use_simulation())}",
        f"ui_edit_hash={_hash_df_in_session('volumes_edit')}",
        f"ui_wide_hash={_hash_df_in_session('volumes_wide')}",
        f"mt_current={_path_mtime_safe((PARQUET_DIR / 'current.parquet')):.0f}",
        f"mt_reswrk={_path_mtime_safe((PARQUET_DIR / 'res_working.parquet')):.0f}",
        f"mt_xlsx={_path_mtime_safe((PREMISSAS_PNL_DIR / 'base_calculos.xlsx')):.0f}",
        "v=1",
    ]
    return hashlib.md5("|".join(pieces).encode()).hexdigest()


# ----------------- validação de dados -----------------
def validate_data_consistency(year: int) -> Dict[str, bool]:
    """
    Valida a consistência dos dados para um ano específico.
    """
    checks = {
        "current_parquet_exists": False,
        "has_realizado_scenario": False,
        "has_re_scenario": False,
        "has_base_calculos": False,
        "has_volume_data": False,
    }

    try:
        from core.models import load_current_long
        from core.paths import BASE_CALCULOS_XLSX

        # Check 1: Current parquet
        df = load_current_long()
        checks["current_parquet_exists"] = not df.empty

        # Check 2: Cenários
        if not df.empty and "cenario" in df.columns:
            scenarios = df["cenario"].unique()
            checks["has_realizado_scenario"] = any("realiz" in str(s).lower() for s in scenarios)
            checks["has_re_scenario"] = any(str(s).upper().startswith("RE") for s in scenarios)

        # Check 3: Base de cálculos
        checks["has_base_calculos"] = BASE_CALCULOS_XLSX.exists()

        # Check 4: Volume data
        vols = resolve_monthly_volumes(year)
        checks["has_volume_data"] = vols is not None and sum(vols.values()) > 0

    except Exception as e:
        print(f"Erro na validação: {e}")

    return checks