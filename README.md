# Simulador DRE (P&L) — MVP com Streamlit

Este projeto é um **esqueleto pronto** para iniciar o simulador de DRE:
- UI web com **Streamlit** (multipáginas).
- Edição de **Volumes por família** (grid editável).
- Exibição do formato de **P&L** a partir do arquivo `data/modelo_pnl.xlsx` (ou fallback para `data/pl_format.xls`).
- Motor de cálculo desacoplado (stub) para futura lógica.

## Estrutura
```
simulador_dre_streamlit/
├─ app.py
├─ core/
│  ├─ calculator.py
│  ├─ io.py
│  ├─ models.py
│  └─ state.py
├─ pages/
│  ├─ 01_Premissas.py
│  ├─ 02_Volumes.py
│  ├─ 03_DRE.py
│  └─ 99_Sobre.py
├─ data/
│  ├─ base_final_longo_para_modelo.xlsx
│  ├─ modelo_pnl.xlsx
│  └─ pl_format.xls
├─ .streamlit/
│  └─ config.toml
├─ requirements.txt
└─ README.md
```

## Requisitos
- Python 3.10+
- Instalar dependências:
```
pip install -r requirements.txt
```

> Observação: para ler `.xls` é necessário **xlrd** (já incluso nas dependências).

## Como rodar
No diretório do projeto:
```
streamlit run app.py
```

## Próximos passos
- Implementar lógica real no `core/calculator.py` (cálculos por linha da DRE, elasticidades, premissas).
- Conectar Redis para persistência temporária de cenários e múltiplos usuários.
- (Opcional) Migrar para FastAPI como backend quando precisar de colaboração em tempo real.
