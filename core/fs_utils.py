# core/fs_utils.py
import os, traceback
import pandas as pd
import streamlit as st

def read_parquet_first_found(candidates):
    last_err = None
    for p in candidates:
        try:
            if os.path.exists(p) and os.path.getsize(p) > 0:
                return pd.read_parquet(p)  # usa pyarrow por padrão quando instalado
        except Exception as e:
            last_err = f"{p}: {e}\n{traceback.format_exc()}"
    if last_err:
        st.warning(f"Falha ao ler Parquet (mostrando último erro):\n\n{last_err}")
    return pd.DataFrame()

def debug_parquet_status():
    paths = [
        "data/parquet/current.parquet",
        "data/current.parquet",
        "data/parquet/res_working.parquet",
        "data/res_working.parquet",
    ]
    rows = []
    for p in paths:
        exists = os.path.exists(p)
        size   = os.path.getsize(p) if exists else 0
        rows.append({"path": p, "exists": exists, "size_bytes": size})
    st.expander("🔍 Diagnóstico dos arquivos", expanded=False).table(rows)
