import streamlit as st

st.set_page_config(page_title="Sobre", page_icon="ℹ️", layout="centered")
st.header("ℹ️ Sobre este MVP")

st.markdown(
    """
    - **Objetivo**: entregar um esqueleto funcional, pronto para receber a lógica de cálculo.
    - **Como evoluir**:
      1. Implementar as regras de cálculo no `core/calculator.py`.
      2. Aplicar premissas (% preço, % custo, impostos) na receita/CPV/Despesas.
      3. Adicionar página de **Cenários** (salvar/duplicar/comparar).
      4. Conectar cache/DB para colaboração multiusuário.
    """
)
