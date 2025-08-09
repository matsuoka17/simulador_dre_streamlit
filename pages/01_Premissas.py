import streamlit as st
from core.state import init_state

st.set_page_config(page_title="Premissas", page_icon="⚙️", layout="wide")
init_state()

st.header("⚙️ Premissas de Simulação (MVP)")

col1, col2, col3 = st.columns(3)
with col1:
    st.session_state.assumptions["delta_preco_pct"] = st.number_input(
        "Variação de Preço (%)", value=0.0, step=0.5, format="%.2f"
    )
with col2:
    st.session_state.assumptions["delta_custo_pct"] = st.number_input(
        "Variação de Custo (%)", value=0.0, step=0.5, format="%.2f"
    )
with col3:
    st.session_state.assumptions["impostos_pct"] = st.number_input(
        "Impostos sobre Receita (%)", value=0.0, step=0.5, format="%.2f"
    )

st.caption("Estes campos são placeholders. A aplicação do efeito ocorrerá quando o motor de cálculo for implementado.")
