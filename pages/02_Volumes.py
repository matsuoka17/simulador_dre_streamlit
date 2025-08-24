# pages/02_Volumes.py
# --------------------------------------------------------------------------------------
# Volumes — Realizado (YTD) travado a partir do CURRENT + Projeção (YTG) editável
# a partir do RES_WORKING. Ocultar YTD, export XLSX, resumo de alterações.
# --------------------------------------------------------------------------------------

import io
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.shared import JsCode

# Core
import core.paths as P
import core.models as M

from core.fs_utils import read_parquet_first_found, debug_parquet_status
debug_parquet_status()

df_current = read_parquet_first_found([
    "data/parquet/current.parquet",
    "data/current.parquet",
])
df_res = read_parquet_first_found([
    "data/parquet/res_working.parquet",
    "data/res_working.parquet",
])


st.set_page_config(page_title="Volumes", page_icon="🧮", layout="wide")

# --------------------------------------------------------------------------------------
# Utilitários
# --------------------------------------------------------------------------------------
@dataclass
class VolumeFormatter:
    MONTH_MAP = M.MONTH_MAP_PT
    REV_MONTH_MAP = {v: k for k, v in MONTH_MAP.items()}
    SCALE_OPTIONS = {"1x": 1, "1.000x": 1_000, "1.000.000x": 1_000_000}

    @classmethod
    def month_cols(cls) -> List[str]:
        return [cls.MONTH_MAP[i] for i in range(1, 13)]

    @classmethod
    def get_scale_factor(cls, label: str) -> int:
        return cls.SCALE_OPTIONS.get(label, 1)

    @classmethod
    def js_value_formatter(cls, scale_factor: int) -> JsCode:
        if scale_factor == 1:
            fmt = "return Math.round(val).toLocaleString('pt-BR');"
        else:
            fmt = "return val.toLocaleString('pt-BR', {minimumFractionDigits:2, maximumFractionDigits:2});"
        return JsCode(f"""
        function(params) {{
          if (params.value === null || params.value === undefined) return '';
          const val = Number(params.value) / {scale_factor};
          {fmt}
        }}""")

    @classmethod
    def js_value_parser(cls, scale_factor: int) -> JsCode:
        return JsCode(f"""
        function(params){{
          var raw = Number((params.oldValue !== undefined ? params.oldValue : params.value) || 0);
          var input = (params.newValue ?? '').toString().trim();
          var SCALE = {scale_factor};
          if (input === '') return raw;

          function brToFloat(s){{
            if (s === null || s === undefined) return NaN;
            s = String(s).trim();
            if (!s) return NaN;
            if (s.indexOf(',') >= 0) s = s.replace(/\\./g,'').replace(/,/g,'.');
            return parseFloat(s);
          }}

          // +10%  /  -5%
          var m = input.match(/^([+-])\\s*([0-9]+[\\.,]?[0-9]*)\\s*%$/);
          if (m){{
            var sign = (m[1] === '+') ? 1 : -1;
            var pct = brToFloat(m[2]);
            if (isNaN(pct)) return raw;
            return Math.max(0, Math.round(raw * (1 + sign * pct/100)));
          }}

          // +1000  /  -250,5
          m = input.match(/^([+-])\\s*([0-9]+(?:[\\.,][0-9]+)?)\\s*$/);
          if (m){{
            var sign = (m[1] === '+') ? 1 : -1;
            var deltaDisp = brToFloat(m[2]);
            if (isNaN(deltaDisp)) return raw;
            return Math.max(0, Math.round(raw + sign * deltaDisp * SCALE));
          }}

          // *1,2
          m = input.match(/^\\*\\s*([0-9]+(?:[\\.,][0-9]+)?)\\s*$/);
          if (m){{
            var factor = brToFloat(m[1]);
            if (isNaN(factor)) return raw;
            return Math.max(0, Math.round(raw * factor));
          }}

          // /2
          m = input.match(/^\\/\\s*([0-9]+(?:[\\.,][0-9]+)?)\\s*$/);
          if (m){{
            var div = brToFloat(m[1]);
            if (!div) return raw;
            return Math.max(0, Math.round(raw / div));
          }}

          // expressão livre ou número absoluto
          var expr = input;
          if (expr.indexOf(',') >= 0) expr = expr.replace(/\\./g,'').replace(/,/g,'.');
          if (/[^0-9+\\-*/().\\s]/.test(expr)){{
            var n = brToFloat(input);
            if (!isNaN(n)) return Math.max(0, Math.round(n * SCALE));
            return raw;
          }}
          try {{
            var valDisp = Function('"use strict"; return (' + expr + ')')();
            if (typeof valDisp !== 'number' || !isFinite(valDisp)) return raw;
            return Math.max(0, Math.round(valDisp * SCALE));
          }} catch(e) {{
            return raw;
          }}
        }}""")

class AppState:
    def __init__(self):
        st.session_state.setdefault("volumes_wide", {})   # {ano: df}
        st.session_state.setdefault("original_wide", {})  # {ano: df original}
        st.session_state.setdefault("grid_version", 0)
        st.session_state.setdefault("hash_map", {})       # {ano: hash}
        st.session_state.setdefault("del_clicks", 0)

    def get(self, year: int, key: str) -> Optional[pd.DataFrame]:
        return st.session_state[key].get(year)

    def set(self, year: int, key: str, df: pd.DataFrame):
        st.session_state[key][year] = df.copy()

    def set_hash(self, year: int, h: int):
        st.session_state["hash_map"][year] = h

    def get_hash(self, year: int) -> Optional[int]:
        return st.session_state["hash_map"].get(year)

    def bump(self):
        st.session_state["grid_version"] = (st.session_state["grid_version"] + 1) % 1000

    def version(self) -> int:
        return st.session_state["grid_version"]

# --------------------------------------------------------------------------------------
# Leitura via models
# --------------------------------------------------------------------------------------
FAM_COL_CAND = ["Família Comercial", "Familia Comercial", "familia_comercial", "Familia"]

def _fam_col(df: pd.DataFrame) -> str:
    for c in FAM_COL_CAND:
        if c in df.columns:
            return c
    return "Família Comercial"

@st.cache_data(show_spinner=False)
def load_current_volume_from_models() -> pd.DataFrame:
    # Realizado (CURRENT) -> long de volumes
    df = M.dre_long(
        cenario_like="Realizado",
        keep_dims=["Família Comercial", "ano", "mes"]
    )
    if df.empty:
        return pd.DataFrame(columns=["Família Comercial","ano","mes","volume"])
    sub = df[df["indicador_id"] == "volume_uc"].copy()
    if sub.empty:
        return pd.DataFrame(columns=["Família Comercial","ano","mes","volume"])
    out = (sub.groupby(["Família Comercial","ano","mes"], dropna=False, as_index=False)["valor"]
              .sum().rename(columns={"valor":"volume"}))
    out["ano"]    = pd.to_numeric(out["ano"], errors="coerce").astype("Int64")
    out["mes"]    = pd.to_numeric(out["mes"], errors="coerce").astype("Int64")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0).astype(float)
    out["Família Comercial"] = out["Família Comercial"].astype(str).str.strip()
    return out

@st.cache_data(show_spinner=False)
def load_res_volume_from_models() -> pd.DataFrame:
    # YTG baseline (editável) -> RES_WORKING
    df = M.res_volume_by_family_long()
    if df.empty:
        return pd.DataFrame(columns=["Família Comercial","ano","mes","volume"])
    df["ano"]    = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")
    df["mes"]    = pd.to_numeric(df["mes"], errors="coerce").astype("Int64")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).astype(float)
    df["Família Comercial"] = df["Família Comercial"].astype(str).str.strip()
    return df

def _latest_year(*dfs: pd.DataFrame) -> Optional[int]:
    years = []
    for d in dfs:
        if d is None or d.empty or "ano" not in d.columns:
            continue
        years.extend(pd.to_numeric(d["ano"], errors="coerce").dropna().astype(int).unique().tolist())
    return max(years) if years else None

def compute_cutoff(df_real: pd.DataFrame, year: int) -> int:
    d = df_real.loc[df_real["ano"] == year]
    if d.empty:
        return 0
    meses = d.groupby("mes", as_index=False)["volume"].sum()
    meses_pos = meses.loc[meses["volume"] > 0, "mes"]
    return int(meses_pos.max()) if not meses_pos.empty else 0

def pivot_year_from_sources(df_real: pd.DataFrame, df_res: pd.DataFrame, year: int) -> Tuple[pd.DataFrame, int]:
    """
    Monta grade wide: Realizado (<=cutoff) travado + RES_WORKING (>cutoff) editável.
    """
    cutoff = compute_cutoff(df_real, year)

    fams = sorted(
        set(df_real.loc[df_real["ano"] == year, "Família Comercial"].unique().tolist())
        | set(df_res.loc[df_res["ano"] == year, "Família Comercial"].unique().tolist())
    )

    def _pivot(df_src: pd.DataFrame) -> pd.DataFrame:
        sub = df_src.loc[df_src["ano"] == year, ["Família Comercial", "mes", "volume"]].copy()
        if sub.empty:
            return pd.DataFrame(columns=["Família Comercial"] + list(range(1, 13)))
        p = sub.pivot_table(index="Família Comercial", columns="mes", values="volume", aggfunc="sum").reindex(
            columns=list(range(1, 13)), fill_value=0
        )
        p = p.reindex(fams).fillna(0)
        p.index.name = "Família Comercial"
        return p.reset_index()

    piv_real = _pivot(df_real)
    piv_res  = _pivot(df_res)
    if piv_res.empty:
        piv_res = pd.DataFrame({"Família Comercial": fams})
        for m in range(1, 13):
            piv_res[m] = 0.0

    final = piv_res.copy()
    for m in range(1, 13):
        if m <= cutoff:
            final[m] = piv_real[m] if m in piv_real.columns else 0.0

    ren = {i: VolumeFormatter.MONTH_MAP[i] for i in range(1, 13)}
    final = final.rename(columns=ren)
    month_cols = VolumeFormatter.month_cols()
    for c in month_cols:
        final[c] = pd.to_numeric(final[c], errors="coerce").fillna(0).round(0).astype(int)
    final["Total Ano"] = final[month_cols].sum(axis=1).astype(int)

    return final, cutoff

def df_hash_for_grid(df: pd.DataFrame) -> int:
    month_cols = VolumeFormatter.month_cols()
    safe_cols = ["Família Comercial"] + [c for c in month_cols if c in df.columns]
    h = pd.util.hash_pandas_object(df[safe_cols], index=False).sum()
    try:
        return int(h)
    except Exception:
        return int(h.astype("int64"))

def calc_changes(original_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    month_cols = VolumeFormatter.month_cols()
    orig_idx = original_df.set_index("Família Comercial")[month_cols]
    curr_idx = current_df.set_index("Família Comercial")[month_cols]
    fams = orig_idx.index.union(curr_idx.index)

    orig_al = orig_idx.reindex(fams).fillna(0).astype(int)
    curr_al = curr_idx.reindex(fams).fillna(0).astype(int)

    orig_long = orig_al.reset_index().melt(id_vars=["Família Comercial"], var_name="Mês", value_name="Antes")
    curr_long = curr_al.reset_index().melt(id_vars=["Família Comercial"], var_name="Mês", value_name="Depois")

    df = orig_long.merge(curr_long, on=["Família Comercial", "Mês"], how="outer").fillna(0)
    df["Δ"] = (df["Depois"] - df["Antes"]).astype(int)
    df = df.loc[df["Δ"] != 0].copy()
    df["Tipo"] = df["Δ"].apply(lambda x: "Inclusão" if x > 0 else "Retirada")
    return df[["Família Comercial", "Mês", "Tipo", "Antes", "Depois", "Δ"]].copy()

# --------------------------------------------------------------------------------------
# ColumnDefs (AgGrid)
# --------------------------------------------------------------------------------------
def build_coldefs_main(scale_factor: int, cutoff: int, hide_realizados: bool) -> list:
    val_fmt = VolumeFormatter.js_value_formatter(scale_factor)
    val_par = VolumeFormatter.js_value_parser(scale_factor)

    def month_col_def(mnum: int) -> dict:
        name = VolumeFormatter.MONTH_MAP[mnum]
        return {
            "field": name,
            "headerName": name,
            "type": ["rightAligned", "numericColumn"],
            "editable": mnum > cutoff,            # Realizado travado
            "valueFormatter": val_fmt.js_code,
            "valueParser": val_par.js_code,
            "sortable": True,
            "cellStyle": JsCode(
                f"""
                function(params){{
                  const locked = {mnum} <= {cutoff};
                  if (locked) return {{ backgroundColor: '#F3F4F6', color: '#6B7280' }};
                  return null;
                }}"""
            ).js_code,
            "hide": (mnum <= cutoff and hide_realizados),
        }

    realizado_children = [month_col_def(m) for m in range(1, cutoff + 1)] if cutoff >= 1 else []
    previsao_children  = [month_col_def(m) for m in range(cutoff + 1, 13)]

    coldefs = [
        {
            "field": "Família Comercial",
            "headerName": "Família",
            "editable": False,
            "suppressMovable": True,
            "cellStyle": JsCode("""
              function(params){
                if (params.data && (params.data["Família Comercial"] === "TOTAL" ||
                                    params.data["Família Comercial"] === "Resultado Operacional")){
                  return {'backgroundColor': '#FFF9C4', 'fontWeight': 'bold'};
                }
                return null;
              }
            """).js_code
        },
    ]

    if realizado_children:
        coldefs.append({
            "headerName": f"Realizado (<= M{cutoff:02d})",
            "marryChildren": True,
            "children": realizado_children
        })
    if previsao_children:
        coldefs.append({
            "headerName": "Previsão (RES)",
            "marryChildren": True,
            "children": previsao_children
        })

    coldefs.append({
        "field": "Total Ano",
        "headerName": "Total Ano",
        "editable": False,
        "type": ["rightAligned", "numericColumn"],
        "valueFormatter": val_fmt.js_code,
        "cellStyle": JsCode("""
          function(params){
            if (params.data && (params.data["Família Comercial"] === "TOTAL" ||
                                params.data["Família Comercial"] === "Resultado Operacional")){
              return {'backgroundColor': '#FFF9C4', 'fontWeight': 'bold'};
            }
            return {'fontWeight': '600'};
          }
        """).js_code
    })
    return coldefs

def build_row_style_main() -> JsCode:
    return JsCode("""
      function(params){
        if (params.node.rowPinned === 'top') {
          return {'backgroundColor': '#FFF9C4', 'fontWeight': 'bold'};
        }
        if (params.data && (params.data["Família Comercial"] === "TOTAL" ||
                            params.data["Família Comercial"] === "Resultado Operacional")){
          return {'backgroundColor': '#FFF9C4', 'fontWeight': 'bold'};
        }
        return null;
      }
    """)

def build_gridopts_main(df: pd.DataFrame, coldefs: list, pinned_totals: dict) -> dict:
    return {
        "columnDefs": coldefs,
        "rowHeight": 34,
        "ensureDomOrder": True,
        "enableRangeSelection": True,
        "stopEditingWhenCellsLoseFocus": True,
        "enterMovesDownAfterEdit": True,
        "singleClickEdit": True,
        "suppressColumnMove": True,
        "pinnedTopRowData": [pinned_totals],
        "getRowStyle": build_row_style_main().js_code,
        "defaultColDef": {"resizable": True},
    }

def build_gridopts_changes(changes_df: pd.DataFrame, scale_factor: int) -> dict:
    val_fmt = VolumeFormatter.js_value_formatter(scale_factor)
    coldefs = [
        {"field": "Família Comercial", "headerName": "Família", "editable": False},
        {"field": "Mês", "editable": False},
        {"field": "Tipo", "editable": False},
        {"field": "Antes", "type": ["rightAligned", "numericColumn"], "valueFormatter": val_fmt.js_code, "editable": False},
        {"field": "Depois", "type": ["rightAligned", "numericColumn"], "valueFormatter": val_fmt.js_code, "editable": False},
        {"field": "Δ", "type": ["rightAligned", "numericColumn"], "valueFormatter": val_fmt.js_code, "editable": False},
        {"field": "DELETE_FLAG", "hide": True},
        {
            "field": "Excluir",
            "headerName": "",
            "editable": False,
            "width": 60,
            "maxWidth": 70,
            "valueGetter": JsCode("function(){return '✖';}").js_code,
            "cellStyle": JsCode("""
              function(){ return {
                display: 'flex', alignItems:'center', justifyContent:'center',
                backgroundColor:'#DC2626', color:'#FFFFFF', fontWeight:'700',
                cursor:'pointer', borderRadius:'4px'
              }; }
            """).js_code,
        },
    ]
    return {
        "columnDefs": coldefs,
        "rowHeight": 36,
        "suppressRowClickSelection": True,
        "stopEditingWhenCellsLoseFocus": True,
        "onCellClicked": JsCode("""
          function(params){
            try{
              if (params.colDef && params.colDef.field === 'Excluir'){
                params.node.setDataValue('DELETE_FLAG', 1);
              }
            }catch(e){}
          }
        """).js_code,
        "defaultColDef": {"resizable": True},
    }

# --------------------------------------------------------------------------------------
# Export XLSX
# --------------------------------------------------------------------------------------
def export_visible_to_xlsx(df_display: pd.DataFrame, visible_months: List[str]) -> bytes:
    cols = ["Família Comercial"] + visible_months + ["Total Ano"]
    to_export = df_display[cols].copy()

    total_row = {"Família Comercial": "TOTAL", **{c: int(to_export[c].sum()) for c in visible_months}, "Total Ano": int(to_export["Total Ano"].sum())}
    out = pd.concat([pd.DataFrame([total_row]), to_export], ignore_index=True)

    bio = io.BytesIO()
    try:
        with pd.ExcelWriter(bio, engine="xlsxwriter") as xw:
            out.to_excel(xw, sheet_name="Volumes", index=False)
            ws = xw.sheets["Volumes"]
            for i, col in enumerate(out.columns):
                maxlen = max(12, out[col].astype(str).map(len).max() if not out.empty else 12)
                ws.set_column(i, i, min(maxlen + 2, 40))
    except Exception:
        with pd.ExcelWriter(bio, engine="openpyxl") as xw:
            out.to_excel(xw, sheet_name="Volumes", index=False)
            ws = xw.sheets["Volumes"]
            from openpyxl.utils import get_column_letter
            for i, col in enumerate(out.columns, start=1):
                maxlen = max(12, out[col].astype(str).map(len).max() if not out.empty else 12)
                ws.column_dimensions[get_column_letter(i)].width = min(maxlen + 2, 40)
    return bio.getvalue()

# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------
def main():
    st.header("🧮 Volumes por Família — Realizado + Projeção (RES)")

    # ---------------- Sidebar (Controles universais) ----------------
    with st.sidebar:
        st.subheader("Controles")
        valid = P.validate_project_structure()
        if not valid["parquet_dir"]:
            st.error("Pasta `data/parquet/` não encontrada.")
        if not valid["current_parquet"]:
            st.error("`data/parquet/current.parquet` não encontrado.")
        if not valid["res_working_parquet"]:
            st.warning("`data/parquet/res_working.parquet` ausente — baseline YTG (RES) ficará vazia.")

        escala_label = st.selectbox("Escala", options=list(VolumeFormatter.SCALE_OPTIONS.keys()), index=0)
        hide_realizados = st.checkbox("Ocultar YTD (Realizado)", value=False)

        st.divider()
        st.caption("Ações sobre a grade:")
        colA, colB = st.columns(2)
        with colA:
            reset_btn = st.button("Recarregar Base", use_container_width=True)
        with colB:
            limpar_btn = st.button("Zerar Projeções (YTG)", use_container_width=True)

    # ---------------- Dados base ----------------
    df_real = load_current_volume_from_models()    # YTD do CURRENT
    df_res  = load_res_volume_from_models()        # YTG baseline do RES_WORKING

    if (df_real.empty and df_res.empty):
        st.error("Sem dados de volumes em CURRENT/RES_WORKING.")
        st.stop()

    year = _latest_year(df_real, df_res)
    if year is None:
        st.error("Não foi possível detectar o ano.")
        st.stop()

    st.caption(f"Ano detectado: **{year}** (último ano com dados)")

    scale = VolumeFormatter.get_scale_factor(escala_label)
    state = AppState()

    original, cutoff = pivot_year_from_sources(df_real, df_res, year)

    # Init/Reset
    if (state.get(year, "original_wide") is None) or reset_btn:
        state.set(year, "original_wide", original)
    if (state.get(year, "volumes_wide") is None) or reset_btn or limpar_btn:
        base = original.copy()
        if limpar_btn and cutoff < 12:
            # Zera somente YTG quando limpar projeções
            month_cols = VolumeFormatter.month_cols()
            for m in range(cutoff + 1, 13):
                col = VolumeFormatter.MONTH_MAP[m]
                if col in base.columns:
                    base[col] = 0
            base["Total Ano"] = base[month_cols].sum(axis=1).astype(int)
        state.set(year, "volumes_wide", base)
        state.set_hash(year, df_hash_for_grid(base))
        state.bump()

    current = state.get(year, "volumes_wide").copy()

    # --- Total YTG (quando Ocultar YTD) ---
    ytg_months = [VolumeFormatter.MONTH_MAP[m] for m in range(cutoff + 1, 13)
                  if VolumeFormatter.MONTH_MAP[m] in current.columns]
    if hide_realizados:
        if ytg_months:
            current["Total YTG"] = current[ytg_months].sum(axis=1, numeric_only=True).fillna(0).astype(int)
        else:
            current["Total YTG"] = 0
    else:
        if "Total YTG" in current.columns:
            current = current.drop(columns=["Total YTG"])

    # Cabeçalhos com totais dos grupos
    real_months = [VolumeFormatter.MONTH_MAP[m] for m in range(1, cutoff + 1)
                   if VolumeFormatter.MONTH_MAP[m] in current.columns]
    prev_months = [VolumeFormatter.MONTH_MAP[m] for m in range(cutoff + 1, 13)
                   if VolumeFormatter.MONTH_MAP[m] in current.columns]

    def _fmt_total_header(val: float, scale_factor: int) -> str:
        disp = float(val) / (scale_factor if scale_factor else 1)
        if scale_factor == 1:
            s = f"{int(round(disp)):,}".replace(",", ".")
        else:
            s = f"{disp:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return s

    real_total = current[real_months].sum(numeric_only=True).sum() if real_months else 0.0
    prev_total = current[prev_months].sum(numeric_only=True).sum() if prev_months else 0.0

    coldefs = build_coldefs_main(scale, cutoff, hide_realizados)
    for grp in coldefs:
        if isinstance(grp, dict) and "children" in grp:
            title = grp.get("headerName", "")
            if title.startswith("Realizado"):
                grp["headerName"] = f"{title} — {_fmt_total_header(real_total, scale)}"
            elif title.startswith("Previsão"):
                grp["headerName"] = f"{title} — {_fmt_total_header(prev_total, scale)}"

    # Pinned totals
    month_cols = VolumeFormatter.month_cols()
    pinned_totals = {
        "Família Comercial": "TOTAL",
        **{c: int(current[c].sum()) for c in month_cols if c in current.columns},
        **({"Total YTG": int(current["Total YTG"].sum())} if "Total YTG" in current.columns else {}),
        "Total Ano": int(current["Total Ano"].sum()),
    }
    gridopts = build_gridopts_main(current, coldefs, pinned_totals)

    # Força reinit do grid quando variáveis-chave mudarem
    main_grid_key = f"main_grid_v{state.version()}_{st.session_state['del_clicks']}_{scale}_{cutoff}_{int(hide_realizados)}"

    grid_resp = AgGrid(
        current,
        gridOptions=gridopts,
        data_return_mode="AS_INPUT",
        update_mode=GridUpdateMode.MODEL_CHANGED,
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,
        theme="balham",
        height=520,
        key=main_grid_key,
    )

    # aplica edições e recalcula total
    try:
        edited = pd.DataFrame(grid_resp["data"]).copy()
        for c in month_cols:
            if c in edited.columns:
                edited[c] = pd.to_numeric(edited[c], errors="coerce").fillna(0).round(0).astype(int)
        edited["Total Ano"] = edited[month_cols].sum(axis=1).astype(int)

        # Recalcular Total YTG após edição
        ytg_months = [VolumeFormatter.MONTH_MAP[m] for m in range(cutoff + 1, 13)
                      if VolumeFormatter.MONTH_MAP[m] in edited.columns]
        if hide_realizados:
            if ytg_months:
                edited["Total YTG"] = edited[ytg_months].sum(axis=1, numeric_only=True).fillna(0).astype(int)
            else:
                edited["Total YTG"] = 0
        else:
            if "Total YTG" in edited.columns:
                edited = edited.drop(columns=["Total YTG"])

        new_hash = df_hash_for_grid(edited)
        old_hash = state.get_hash(year)
        if (old_hash is None) or (new_hash != old_hash):
            state.set(year, "volumes_wide", edited)
            state.set_hash(year, new_hash)
            state.bump()
            st.rerun()
    except Exception:
        pass

    current = state.get(year, "volumes_wide").copy()
    original = state.get(year, "original_wide").copy()

    # ---------------- Resumo/Alterações ----------------
    st.markdown("### 🔁 Alterações aplicadas (vs base carregada)")
    changes = calc_changes(original, current)
    if changes.empty:
        st.info("Nenhuma alteração em relação à base original.")
    else:
        changes = changes.copy()
        changes["DELETE_FLAG"] = 0
        changes["Excluir"] = "✖"

        ch_opts = build_gridopts_changes(changes, scale)
        ch_resp = AgGrid(
            changes,
            gridOptions=ch_opts,
            data_return_mode="AS_INPUT",
            update_mode=GridUpdateMode.MODEL_CHANGED,
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True,
            theme="balham",
            height=360,
            key=f"changes_grid_v{state.version()}_{st.session_state['del_clicks']}"
        )

        try:
            ch_after = pd.DataFrame(ch_resp["data"])
            to_del = ch_after.loc[ch_after.get("DELETE_FLAG", 0) == 1]
            if not to_del.empty:
                work = current.copy()
                base = original.set_index("Família Comercial")
                for _, row in to_del.iterrows():
                    fam = row["Família Comercial"]
                    m   = row["Mês"]
                    if fam in base.index and m in base.columns:
                        val = int(base.loc[fam, m])
                        work.loc[work["Família Comercial"] == fam, m] = val
                work["Total Ano"] = work[month_cols].sum(axis=1).astype(int)
                state.set(year, "volumes_wide", work)
                state.set_hash(year, df_hash_for_grid(work))
                st.session_state["del_clicks"] += 1
                state.bump()
                st.rerun()
        except Exception:
            pass

    # ---------------- Export XLSX ----------------
    st.markdown("### 💾 Exportar")
    visibles = []
    for m in range(1, 13):
        name = VolumeFormatter.MONTH_MAP[m]
        if not (m <= cutoff and hide_realizados):
            visibles.append(name)

    xlsx_bytes = export_visible_to_xlsx(current, visibles)
    st.download_button(
        label="⬇️ Baixar .xlsx",
        data=xlsx_bytes,
        file_name=f"volumes_{year}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.markdown("---")
    st.caption("YTD (Realizado) vem do CURRENT; YTG (Projeção) parte de RES_WORKING e é editável no grid.")

# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
