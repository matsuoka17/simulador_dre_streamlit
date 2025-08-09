import pandas as pd
from pathlib import Path

def load_volume_base(path: str | Path) -> pd.DataFrame:
    """Carrega a base de volumes (xlsx). Espera colunas: 
    ['SKU','Produto','Família Comercial','ano','mes','volume'].
    """
    path = Path(path)
    df = pd.read_excel(path)
    # sanity check básico
    expected_cols = {"SKU","Produto","Família Comercial","ano","mes","volume"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colunas faltantes na base de volume: {missing}")
    return df

def load_pl_format(path: str | Path) -> pd.DataFrame:
    """Carrega o formato da DRE (xls/xlsx). 
    Não impõe esquema fixo — exibe o que vier para referenciar as linhas.
    """
    path = Path(path)
    # pandas escolhe engine pelo sufixo; xlrd é necessário para .xls
    df = pd.read_excel(path)
    return df
