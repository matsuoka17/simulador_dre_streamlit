import streamlit as st
from core.state import init_state

st.set_page_config(page_title="Simulador P&L", page_icon="📊", layout="wide")

# Insere logo Leao
from core.ui_branding import inject_sidebar_logo_bottom
inject_sidebar_logo_bottom("models/logo.jpeg", max_width_px=180)


init_state()

st.title("📊 Simulador P&L")
st.markdown(
    """
    Este é o simulador de P&L baseado em volume. Navegue pelas páginas à esquerda:
   
    - **Volumes**: edite volumes por família/mês diretamente na aba para obter os reflexos tanto no P&L quando nos gráficos.
    
    - **P&L**: visualize o formato do P&L e um resultado provisório do ano.
    
    - **Dashboard**: Análise gráfica de alterações de volume.
    
    """
)

st.info("Dica: você pode e deve alterar os valores diretamente na tabela dentro da página 'Volumes'.")
