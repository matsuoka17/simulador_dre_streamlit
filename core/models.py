from pydantic import BaseModel, Field
from typing import Dict, Any

class Assumptions(BaseModel):
    delta_preco_pct: float = Field(default=0.0, description="Variação de preço (%)")
    delta_custo_pct: float = Field(default=0.0, description="Variação de custo (%)")
    impostos_pct: float = Field(default=0.0, description="Impostos sobre receita (%)")

class CalcPayload(BaseModel):
    assumptions: Assumptions
    # estruturas minimamente necessárias — podem ser refinadas depois
