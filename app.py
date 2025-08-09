import streamlit as st
from core.state import init_state

st.set_page_config(page_title="Simulador DRE (MVP)", page_icon="📊", layout="wide")
init_state()

st.title("📊 Simulador P&L")
st.markdown(
    """
    Este é o *MVP* do simulador de DRE. Navegue pelas páginas à esquerda:
    - **Premissas**: defina parâmetros de simulação (placeholders).
    - **Volumes**: edite volumes por família/mês/ano.
    - **DRE**: visualize o formato da DRE e um resultado provisório (zeros).
    
    > A lógica de cálculo será conectada no próximo passo (arquivo `core/calculator.py`).
    """
)

st.info("Dica: você pode colar valores diretamente no grid de edição em 'Volumes'.")
st.caption("Build inicial — estruturado para evoluir com motor de cálculo e persistência de cenários.")
