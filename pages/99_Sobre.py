import streamlit as st

st.set_page_config(page_title="Sobre", page_icon="ℹ️", layout="centered")

# Insere logo Leao
from core.ui_branding import inject_sidebar_logo_bottom
inject_sidebar_logo_bottom("models/logo.jpeg", max_width_px=180)


st.header("ℹ️ Projeção e análise de P&L")

st.markdown(
    """
    - Objetivo: Permitir as análises de mudanças de volumes vindas do Passo 2 para representação no P&L no Passo 4.
    
    **- O que já está pronto?**
    
      1. Bases de realizado, BP e RE's oficiais, diretamente do Qlik.
      
      2. Na aba volumes, já trazemos os números propostos no Passo 2, com poder de alteração dentro da mesma página, para simulação de P&L.
      
      3. P&L de previsibilidade dos números, com possibilidades comparativas entra BP, Realizado, RE's vs Projeção.
      
      4. Reports podem ser extraídos para Excel.
    """)

   # Separador com respiro (substitui os dois <div> e o <hr>)
st.markdown(
    "<hr style='border:0; border-top:1px solid #E5E7EB; margin: 32px 0 32px 0;'/>",
    unsafe_allow_html=True)

st.markdown(
    """
        
    **- Como vamos evoluir?**
    
      1. Armazenagem dentro do modelo, das simulações.
      
      2. Indicadores que auxiliam a tomada de decisão.
      
      3. Modelos de Machine Learning para análises preditivas.
      
      4. Modelo de LLM para interação e auxílio na tomada de decisão.
    """
)