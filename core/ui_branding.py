# core/ui_branding.py
from __future__ import annotations
from pathlib import Path
import base64
import streamlit as st

from core.sim import (
    ensure_global_defaults,
    render_global_scale_selector,
    render_global_scenario_selector,
    render_global_sim_toggle,
)


def _inject_sidebar_css_once():
    if st.session_state.get("_brand_css_injected"):
        return
    st.markdown(
        """
        <style>
          [data-testid="stSidebar"] > div:first-child { position: relative; min-height: 100vh; }
          [data-testid="stSidebar"] .sidebar-logo {
            position: absolute; left: 12px; right: 12px; bottom: 16px;
            padding-top: 8px; background: transparent;
          }
          [data-testid="stSidebar"] .sidebar-logo img {
            display: block; margin: 0 auto; width: 100%; max-width: 180px; opacity: 0.98;
          }
          [data-testid="stSidebar"] .sidebar-spacer { height: 72px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state["_brand_css_injected"] = True


def inject_sidebar_logo_bottom(img_path: str = None, max_width_px: int = 180) -> None:
    """CORRIGIDO: usar paths absolutos e fallback para logo padrão."""
    from core.paths import LOGO_PATH, PROJECT_ROOT

    _inject_sidebar_css_once()

    # Usar path padrão se não especificado
    if img_path is None:
        p = LOGO_PATH
    else:
        p = Path(img_path)
        # Se relativo, resolver a partir da raiz do projeto
        if not p.is_absolute():
            p = PROJECT_ROOT / img_path

    if not p.exists():
        st.sidebar.markdown("<div class='sidebar-spacer'></div>", unsafe_allow_html=True)
        return

    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    st.sidebar.markdown("<div class='sidebar-spacer'></div>", unsafe_allow_html=True)
    st.sidebar.markdown(
        f"""
        <div class="sidebar-logo">
            <img alt="Logo" src="data:{mime};base64,{b64}" style="max-width:{int(max_width_px)}px;" />
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand_and_controls(
        *,
        show_sim_toggle: bool = True,
        default_use_sim: bool = True,
        show_scale_selector: bool = True,
        default_scale: str = "1x",
        show_scenario_selector: bool = True,
        default_scenario: str = "Realizado (YTD+YTG)",
        logo_path: str = None,  # CORRIGIDO: None para usar padrão do paths.py
        logo_max_width_px: int = 180,
) -> None:
    """Sidebar padrão com branding + controles globais."""
    ensure_global_defaults(
        default_scenario=default_scenario,
        default_use_sim=default_use_sim,
        default_scale=default_scale,
    )
    with st.sidebar:
        st.markdown("### Controles Globais")
        if show_scenario_selector:
            render_global_scenario_selector(sidebar=False, label="Cenário")
        if show_scale_selector:
            render_global_scale_selector(sidebar=False, label="Escala")
        if show_sim_toggle:
            render_global_sim_toggle(label="Usar Projeção (volumes YTG)", sidebar=False)
        st.markdown("---")
    inject_sidebar_logo_bottom(img_path=logo_path, max_width_px=logo_max_width_px)