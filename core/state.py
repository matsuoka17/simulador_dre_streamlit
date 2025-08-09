import streamlit as st

DEFAULT_ASSUMPTIONS = {
    "delta_preco_pct": 0.0,
    "delta_custo_pct": 0.0,
    "impostos_pct": 0.0,
}

def init_state():
    if "assumptions" not in st.session_state:
        st.session_state["assumptions"] = DEFAULT_ASSUMPTIONS.copy()
    if "volumes_edit" not in st.session_state:
        st.session_state["volumes_edit"] = None
    if "dre_results" not in st.session_state:
        st.session_state["dre_results"] = None
