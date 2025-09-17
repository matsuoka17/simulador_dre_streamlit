# teste_pnl.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULE_IMPORT_ERROR: Exception | None = None
try:
    import core.paths as P
    import core.models as M
    import core.pnl_gateway as G
    import core.sim as S
except Exception as exc:  # pragma: no cover - exibido apenas na UI
    MODULE_IMPORT_ERROR = exc
    P = M = G = S = None  # type: ignore[assignment]


def _scale_factor(scale: str) -> int:
    return {"1x": 1, "1.000x": 1_000, "1.000.000x": 1_000_000}[scale]


def _format_br(x, casas: int = 2):
    try:
        val = float(x)
    except Exception:
        return x
    s = f"{val:,.{casas}f}"  # 1,234,567.89
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s


def _apply_scale_and_format(df: pd.DataFrame, scale: str, casas: int = 2) -> pd.DataFrame:
    factor = _scale_factor(scale)
    num_cols = [c for c in df.columns if c != "Conta"]
    out = df.copy()
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0) / factor
        out[c] = out[c].map(lambda v: _format_br(v, casas=casas))
    return out


def main() -> None:
    st.set_page_config(page_title="Teste P&L", layout="wide")
    st.title("Validador do P&L • Smoke Test")

    if MODULE_IMPORT_ERROR:
        st.error(f"❌ Falha ao importar módulos core: {MODULE_IMPORT_ERROR}")
        st.stop()

    with st.sidebar:
        st.header("Configuração")
        valid = P.validate_project_structure()
        if not valid["current_parquet"]:
            st.error("Arquivo `data/parquet/current.parquet` não encontrado.")
        if not valid["parquet_dir"]:
            st.error("Pasta `data/parquet/` não encontrada.")
        if not valid["base_calculos_xlsx"]:
            st.warning("Planilha `data/premissas_pnl/base_calculos.xlsx` ausente (fallback RE será usado no YTG).")

        try:
            curr = M.load_current_long()
            anos = sorted(pd.to_numeric(curr["ano"], errors="coerce").dropna().astype(int).unique().tolist())
        except Exception:
            anos = []
        year = st.selectbox("Ano", anos, index=len(anos) - 1 if anos else 0, disabled=not anos)

        scen_label = st.selectbox("Cenário", ["BP (FY)", "RE (FY)", "Realizado (YTD+YTG)"], index=2)
        scale_label = st.selectbox("Escala visual", ["1x", "1.000x", "1.000.000x"], index=0)

        use_sim = False
        if "Realizado" in scen_label:
            use_sim = st.toggle("Usar Projeção (YTG)", value=False, help="OFF→ Grupo 3 (RES). ON→ Grupo 4 (UI).")

    st.session_state[S._SCEN_LABEL_KEY] = scen_label
    st.session_state[S._SCALE_LABEL_KEY] = scale_label
    st.session_state[S._USE_SIM_KEY] = use_sim

    with st.expander("🔎 Diagnóstico", expanded=False):
        valid = P.validate_project_structure()
        st.write("Estrutura:", {k: str(v) if isinstance(v, Path) else v for k, v in valid.items()})
        try:
            last_re = M.find_latest_re_scenario(year if year else None)
            st.write("Último RE detectado:", last_re)
        except Exception as e:
            st.warning(f"RE não detectado: {e}")
        try:
            cut = M.realized_cutoff_by_year("Realizado")
            st.write("Cutoff Realizado:", cut)
        except Exception as e:
            st.warning(f"Cutoff não calculado: {e}")

    if not anos:
        st.stop()

    try:
        df = G.get_pnl_for_current_settings(year=year, scenario_label=scen_label, use_projection=use_sim)
    except Exception as e:
        st.error(f"❌ Erro ao montar o P&L: {e}")
        st.stop()

    required_cols = ["Conta"] + M.MONTHS_PT + ["Total Ano"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Contrato inválido. Faltando colunas: {missing}")
        st.dataframe(df.head(50), use_container_width=True)
        st.stop()

    if "Realizado" in scen_label and not use_sim:
        if all(float(df[c].sum()) == 0.0 for c in M.MONTHS_PT + ["Total Ano"]):
            st.warning("Toggle OFF (Grupo 3) retornou zeros. Verifique se o `res_working.parquet` tem dados para o ano selecionado.")

    st.subheader(f"P&L  {scen_label} • {year}")
    st.caption("Escala e formatação aplicadas apenas para exibição visual (padrão brasileiro).")

    df_fmt = _apply_scale_and_format(df, scale_label, casas=2)
    st.dataframe(df_fmt, use_container_width=True, height=560)

    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        try:
            idx = df.set_index("Conta")
            receita = float(idx.loc["receita_bruta", "Total Ano"]) if "receita_bruta" in idx.index else 0.0
            rl = float(idx.loc["receita_liquida", "Total Ano"]) if "receita_liquida" in idx.index else 0.0
            ebitda = float(idx.loc["ebitda", "Total Ano"]) if "ebitda" in idx.index else 0.0
            vol = float(idx.loc["volume_uc", "Total Ano"]) if "volume_uc" in idx.index else 0.0
        except Exception:
            receita = rl = ebitda = vol = 0.0

        f = _scale_factor(scale_label)
        c1.metric("Receita Bruta (Total Ano)", _format_br(receita / f))
        c2.metric("Receita Líquida (Total Ano)", _format_br(rl / f))
        c3.metric("EBITDA (Total Ano)", _format_br(ebitda / f))
        c4.metric("Volume UC (Total Ano)", _format_br(vol / f))


if __name__ == "__main__":  # pragma: no cover - executado apenas via streamlit
    main()


def test_projection_volumes_override(monkeypatch):
    """
    Garante que o gateway respeita os volumes editados da UI quando a projeção está ativa.
    """
    import pytest
    import core.calculator as C

    if MODULE_IMPORT_ERROR:
        pytest.skip("Módulos core indisponíveis para o teste de projeção")

    year = 2025
    month_cols = M.MONTHS_PT
    base_row = {col: 10 for col in month_cols}
    base_row["Total Ano"] = sum(base_row.values())
    piv_real = pd.DataFrame([{**{"Conta": "volume_uc"}, **base_row}])

    def fake_dre_matrix_total(year: int, cenario_like: str = "Realizado") -> pd.DataFrame:
        return piv_real.copy()

    def fake_realized_cutoff_by_year(_: str) -> Dict[int, int]:
        return {year: 6}

    def fake_load_current_long() -> pd.DataFrame:
        return pd.DataFrame({"ano": [year], "mes": [1], "indicador_id": ["volume_uc"], "valor": [10], "cenario": ["Realizado"]})

    def fake_find_latest_re_scenario(_: int) -> str:
        return "RE 01.25"

    def fake_res_volume_by_family_long() -> pd.DataFrame:
        return pd.DataFrame({
            "Família Comercial": ["Fam Base"],
            "ano": [year],
            "mes": [7],
            "volume": [50.0],
        })

    monkeypatch.setattr(M, "dre_matrix_total", fake_dre_matrix_total)
    monkeypatch.setattr(M, "realized_cutoff_by_year", fake_realized_cutoff_by_year)
    monkeypatch.setattr(M, "load_current_long", fake_load_current_long)
    monkeypatch.setattr(M, "find_latest_re_scenario", fake_find_latest_re_scenario)
    monkeypatch.setattr(M, "res_volume_by_family_long", fake_res_volume_by_family_long)

    captured: Dict[str, object] = {}

    def fake_build_simulado_pivot(*, df, piv_real, year, cutoff, base_calc_path, volumes_edit, volumes_res,
                                   volume_mode, dme_pct, ui_month_totals, conv_source):
        captured["volume_mode"] = volume_mode
        captured["volumes_edit"] = volumes_edit
        captured["ui_month_totals"] = ui_month_totals
        base = piv_real.set_index("Conta")
        out = base.copy()
        for idx in out.index:
            if idx == "volume_uc":
                for m in range(1, 13):
                    col = month_cols[m - 1]
                    if m > cutoff:
                        out.loc[idx, col] = ui_month_totals.get(m, 0) if ui_month_totals else 0
                    else:
                        out.loc[idx, col] = base.loc[idx, col]
                out.loc[idx, "Total Ano"] = int(sum(out.loc[idx, col] for col in month_cols))
        return out.reset_index()

    monkeypatch.setattr(C, "build_simulado_pivot", fake_build_simulado_pivot)

    st.session_state.clear()
    st.session_state[S._SCEN_LABEL_KEY] = "Realizado (YTD+YTG)"
    st.session_state[S._USE_SIM_KEY] = True
    st.session_state[S._SCALE_LABEL_KEY] = "1x"

    ui_df = pd.DataFrame([
        {"Família Comercial": "Fam UI", "ano": year, "mes": 7, "volume": 200.0},
        {"Família Comercial": "Fam UI", "ano": year, "mes": 8, "volume": 300.0},
    ])
    st.session_state["volumes_edit"] = ui_df

    try:
        result = G.get_pnl_for_current_settings(
            year=year,
            scenario_label="Realizado (YTD+YTG)",
            use_sim=True,
        )
    finally:
        st.session_state.clear()

    assert captured["volume_mode"] == "ui"
    assert isinstance(captured["volumes_edit"], pd.DataFrame)
    assert set(captured["volumes_edit"].columns) == {"Família Comercial", "ano", "mes", "volume"}
    assert captured["ui_month_totals"][7] == 200
    assert captured["ui_month_totals"][8] == 300

    idx = result.set_index("Conta")
    assert idx.loc["volume_uc", "Jul"] == 200
    assert idx.loc["volume_uc", "Ago"] == 300
    assert idx.loc["volume_uc", "Jun"] == 10
