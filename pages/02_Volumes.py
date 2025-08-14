# pages/02_Volumes.py
# --------------------------------------------------------------------------------------
# Módulo: Volumes (Simulador DRE)
#
# Objetivos:
# - Edição de volumes (Família x Meses), com TOTAL fixo no topo e exibição com escala
#   (1x / 1.000x / 1.000.000x). Armazenamento SEM escala (unidades originais).
# - Tabela “De → Para” para visualizar alterações vs base original do Excel.
# - Exclusão imediata de uma alteração (clique no “X”) com reversão do valor na tabela
#   principal, sem loops de rerun.
# - INPUT tipo “Excel”: você pode digitar fórmulas simples diretamente na célula:
#     • Relativo: +100, -350, *1,2, /2, +10%, -7,5%
#     • Absoluto “light”: 1000-350, (200*3)+50, 1500/3  (com sanitização)
#
# Estratégias de robustez:
# 1) Anti-loop & Anti-race:
#    - Clique no X enfileira a reversão em st.session_state["pending_reverts"] e
#      faz st.rerun(). No PRÓXIMO run:
#         * Aplicamos reversões ANTES de montar os grids (one-shot).
#         * Marcamos just_reverted=True => não sobrescrevemos o state com o dataset
#           retornado pelo AgGrid neste mesmo run (pode estar stale).
#    - Forçamos remount do grid principal após reverter (vol_grid_ver++) para que o
#      AgGrid descarte buffers internos e use o rowData atualizado do state.
#    - A tabela “De→Para” também tem key com versão (mov_grid_ver) para resetar estado.
# 2) Commit de edição confiável no grid principal:
#    - stopEditingWhenCellsLoseFocus, enterMovesDownAfterEdit, singleClickEdit
#    - update_mode=MODEL_CHANGED
#
# Pré-requisitos:
#   - streamlit-aggrid==1.1.7
#   - Funções utilitárias: core.state.init_state, core.io.load_volume_base
# --------------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
from pathlib import Path
from core.state import init_state
from core.io import load_volume_base

# AgGrid
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from st_aggrid.shared import JsCode

# --------------------------------------------------------------------------------------
# Setup inicial
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Volumes", page_icon="🧮", layout="wide")
init_state()
st.header("🧮 Volumes por Família — meses em colunas + TOTAL no topo")

# Versões dos grids (para forçar remount quando necessário)
st.session_state.setdefault("mov_grid_ver", 0)  # grid “De→Para”
st.session_state.setdefault("vol_grid_ver", 0)  # grid principal

toolbar = st.container()  # barra superior (botões de ações)

# --------------------------------------------------------------------------------------
# Carregamento da base original (Excel) e sanitização
# --------------------------------------------------------------------------------------
data_path = Path("data/base_final_longo_para_modelo.xlsx")
uploaded = None
if not data_path.exists():
    st.warning("Base de volumes não encontrada em 'data/base_final_longo_para_modelo.xlsx'. Faça o upload abaixo.")
    uploaded = st.file_uploader("Enviar base de volumes (.xlsx)", type=["xlsx"], accept_multiple_files=False, key="upload_volumes")
    if uploaded is None:
        st.info("Se preferir, coloque o arquivo depois na pasta 'data/' com o nome 'base_final_longo_para_modelo.xlsx'.")
        st.stop()
    base_df = load_volume_base(uploaded)
else:
    base_df = load_volume_base(data_path)

# Tipos
base_df["ano"] = pd.to_numeric(base_df["ano"], errors="coerce").astype("Int64")
base_df["mes"] = pd.to_numeric(base_df["mes"], errors="coerce").astype("Int64")
base_df["volume"] = pd.to_numeric(base_df["volume"], errors="coerce").fillna(0).astype(float)

anos = sorted([int(a) for a in base_df["ano"].dropna().unique().tolist()])
if not anos:
    st.error("A coluna 'ano' não possui valores válidos.")
    st.stop()

# --------------------------------------------------------------------------------------
# Filtros superiores (Ano + Escala de exibição)
# --------------------------------------------------------------------------------------
c_top1, c_top2, _ = st.columns([1, 1.2, 5])
with c_top1:
    ano_sel = st.selectbox("Ano", options=anos, index=len(anos) - 1, help="Selecione o ano para editar os volumes.")
with c_top2:
    escala_lbl = st.selectbox("Escala de exibição", options=["1x", "1.000x", "1.000.000x"], index=0)
scale_factor = {"1x": 1, "1.000x": 1_000, "1.000.000x": 1_000_000}[escala_lbl]

# --------------------------------------------------------------------------------------
# Monta pivot ORIGINAL (Excel) para o ano selecionado
# --------------------------------------------------------------------------------------
df_y = base_df.loc[base_df["ano"] == ano_sel].copy()
if df_y.empty:
    st.warning(f"Não há dados para o ano {ano_sel}.")
    st.stop()

map_m = {1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun", 7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"}
meses_num = list(range(1, 13))
month_cols = [map_m[m] for m in meses_num]

agg = df_y.groupby(["Família Comercial", "mes"], as_index=False)["volume"].sum()
orig_pivot = (
    agg.pivot(index="Família Comercial", columns="mes", values="volume")
       .reindex(columns=meses_num, fill_value=0)
       .fillna(0)
)
orig_pivot.columns = [map_m.get(int(c), str(c)) for c in orig_pivot.columns]
orig_pivot = orig_pivot.reset_index()

# ORIGINAL wide (para comparar/reverter)
orig_wide = orig_pivot[["Família Comercial"] + month_cols].copy()
for c in month_cols:
    orig_wide[c] = pd.to_numeric(orig_wide[c], errors="coerce").fillna(0).round(0).astype(int)
orig_wide["Total Ano"] = orig_wide[month_cols].sum(axis=1).astype(int)

# --------------------------------------------------------------------------------------
# One-shot: aplica reversões pendentes deste ano (geradas pelo clique no X)
# e marca que NÃO devemos “gravar de volta” o dataset do grid neste run.
# --------------------------------------------------------------------------------------
just_reverted = False
if "pending_reverts" in st.session_state:
    ops = [op for op in st.session_state["pending_reverts"] if op.get("ano") == ano_sel]
    if ops:
        st.session_state.setdefault("volumes_wide", {})
        if ano_sel not in st.session_state["volumes_wide"]:
            st.session_state["volumes_wide"][ano_sel] = orig_wide[["Família Comercial"] + month_cols + ["Total Ano"]].copy()

        wide = st.session_state["volumes_wide"][ano_sel].copy()
        for op in ops:
            fam = op.get("fam"); mes_nome = op.get("mes_nome")
            if (fam in wide["Família Comercial"].values) and (mes_nome in month_cols):
                orig_val = int(orig_wide.loc[orig_wide["Família Comercial"] == fam, mes_nome].values[0])
                wide.loc[wide["Família Comercial"] == fam, mes_nome] = orig_val

        wide["Total Ano"] = wide[month_cols].sum(axis=1).astype(int)
        st.session_state["volumes_wide"][ano_sel] = wide
        just_reverted = True

        # Força remount do grid principal no próximo render (descarta buffers internos)
        st.session_state["vol_grid_ver"] += 1

    # Limpa operações do ano corrente (mantém de outros anos)
    st.session_state["pending_reverts"] = [op for op in st.session_state["pending_reverts"] if op.get("ano") != ano_sel]

# --------------------------------------------------------------------------------------
# Base EDITÁVEL (state) — se houver, usa; senão, começa da original
# --------------------------------------------------------------------------------------
if "volumes_wide" in st.session_state and ano_sel in st.session_state["volumes_wide"]:
    pivot_base = st.session_state["volumes_wide"][ano_sel][["Família Comercial"] + month_cols].copy()
else:
    pivot_base = orig_wide[["Família Comercial"] + month_cols].copy()

# Inteiros + total
for c in month_cols:
    pivot_base[c] = pd.to_numeric(pivot_base[c], errors="coerce").fillna(0).round(0).astype(int)
pivot_base["Total Ano"] = pivot_base[month_cols].sum(axis=1).astype(int)

# Linha TOTAL fixa para o topo
totals_dict = {"Família Comercial": "TOTAL", **{c: int(pivot_base[c].sum()) for c in month_cols}, "Total Ano": int(pivot_base["Total Ano"].sum())}

# --------------------------------------------------------------------------------------
# JS: formatadores/parsers com ESCALA (exibição ≠ armazenamento)
#  - Formatter divide pelo fator (visual)
#  - Parser calcula fórmulas e multiplica pelo fator (grava raw)
# --------------------------------------------------------------------------------------
if scale_factor == 1:
    vf_js_body = "return Math.round(val).toLocaleString('pt-BR');"
else:
    vf_js_body = "return val.toLocaleString('pt-BR', {minimumFractionDigits:2, maximumFractionDigits:2});"

value_formatter = JsCode(f"""
function(params) {{
  if (params.value === null || params.value === undefined) return '';
  const val = Number(params.value) / {scale_factor};
  {vf_js_body}
}}
""")

# Parser com “modo Excel” (relativo + absoluto “light”), mínimo 0 e inteiro
value_parser_formula_js = """
function(params) {
  // Valor atual (raw, sem escala). oldValue costuma ser mais confiável aqui.
  var rawCurrent = Number((params.oldValue !== undefined ? params.oldValue : params.value) || 0);
  var input = (params.newValue ?? '').toString().trim();

  // Se vazio, mantém o valor atual
  if (input === '') return rawCurrent;

  var SCALE = """ + str(scale_factor) + """;

  // Converte "br" -> float (aceita "." como milhar e "," como decimal)
  function brToFloat(s) {
    if (s === null || s === undefined) return NaN;
    s = String(s).trim();
    if (s === '') return NaN;
    if (s.indexOf(',') >= 0) {
      s = s.replace(/\./g,'').replace(/,/g,'.');
    }
    return parseFloat(s);
  }

  var m;

  // 1) Operações RELATIVAS com percentuais: +10%  /  -7,5%
  m = input.match(/^([+-])\s*([0-9]+[\.,]?[0-9]*)\s*%$/);
  if (m) {
    var sign = (m[1] === '+') ? 1 : -1;
    var pct  = brToFloat(m[2]) / 100.0;
    if (isNaN(pct)) return rawCurrent;
    var newRaw = Math.round(rawCurrent * (1 + sign * pct));
    if (newRaw < 0) newRaw = 0;
    return newRaw;
  }

  // 2) Operações RELATIVAS aditivas: +N  /  -N  (N na escala visível)
  m = input.match(/^([+-])\s*([0-9]+(?:[\.,][0-9]+)?)\s*$/);
  if (m) {
    var sign = (m[1] === '+') ? 1 : -1;
    var deltaDisp = brToFloat(m[2]);
    if (isNaN(deltaDisp)) return rawCurrent;
    var newRaw = Math.round(rawCurrent + sign * deltaDisp * SCALE);
    if (newRaw < 0) newRaw = 0;
    return newRaw;
  }

  // 3) Multiplicação RELATIVA: *fator
  m = input.match(/^\*\s*([0-9]+(?:[\.,][0-9]+)?)\s*$/);
  if (m) {
    var factor = brToFloat(m[1]);
    if (isNaN(factor)) return rawCurrent;
    var newRaw = Math.round(rawCurrent * factor);
    if (newRaw < 0) newRaw = 0;
    return newRaw;
  }

  // 4) Divisão RELATIVA: /divisor
  m = input.match(/^\/\s*([0-9]+(?:[\.,][0-9]+)?)\s*$/);
  if (m) {
    var div = brToFloat(m[1]);
    if (!div) return rawCurrent;
    var newRaw = Math.round(rawCurrent / div);
    if (newRaw < 0) newRaw = 0;
    return newRaw;
  }

  // 5) Expressão ABSOLUTA “light” (na escala visível)
  //    Ex.: 1000-350, (200*3)+50, 1500/3
  //    Sanitização: permite apenas 0-9 + - * / . , ( ) e espaços.
  //    Percentual em expressão absoluta não é aceito (apenas relativo).
  var expr = input;
  if (expr.indexOf(',') >= 0) {
    expr = expr.replace(/\./g,'').replace(/,/g,'.'); // BR -> EN
  }
  if (/[^0-9+\-*/().\s]/.test(expr)) {
    // fallback: tenta número único na escala visível
    var n = brToFloat(input);
    if (!isNaN(n)) {
      var newRaw = Math.round(n * SCALE);
      if (newRaw < 0) newRaw = 0;
      return newRaw;
    }
    return rawCurrent;
  }

  try {
    var valDisp = Function('"use strict"; return (' + expr + ')')();
    if (typeof valDisp !== 'number' || !isFinite(valDisp)) return rawCurrent;
    var newRaw = Math.round(valDisp * SCALE);
    if (newRaw < 0) newRaw = 0;
    return newRaw;
  } catch(e) {
    return rawCurrent;
  }
}
"""
value_parser_int = JsCode(value_parser_formula_js)

get_row_style = JsCode("""
function(params) {
  if (params.node.rowPinned === 'top' || (params.data && params.data["Família Comercial"] === "TOTAL")) {
    return {'backgroundColor': '#374151','color': '#F9FAFB','fontWeight': 'bold'};
  }
  return null;
}
""")

# --------------------------------------------------------------------------------------
# GRID PRINCIPAL (edição)
# --------------------------------------------------------------------------------------
gb = GridOptionsBuilder.from_dataframe(pivot_base)
gb.configure_column("Família Comercial", header_name="Família", editable=False)
gb.configure_column("Total Ano", type=["rightAligned","numericColumn"], valueFormatter=value_formatter, editable=False)
for col in month_cols:
    gb.configure_column(col, type=["rightAligned","numericColumn"], valueFormatter=value_formatter, valueParser=value_parser_int, editable=True)

gb.configure_grid_options(
    pinnedTopRowData=[totals_dict],
    getRowStyle=get_row_style,
    rowHeight=34,
    ensureDomOrder=True,
    enableRangeSelection=True,
    # Evita perder edição no rerun do Streamlit
    stopEditingWhenCellsLoseFocus=True,
    enterMovesDownAfterEdit=True,
    singleClickEdit=True,
)

grid_resp = AgGrid(
    pivot_base,
    gridOptions=gb.build(),
    data_return_mode="AS_INPUT",
    update_mode=GridUpdateMode.MODEL_CHANGED,  # captura mudanças no modelo
    fit_columns_on_grid_load=True,
    allow_unsafe_jscode=True,
    theme="balham",
    height=520,
    key=f"grid_volumes_v{st.session_state['vol_grid_ver']}"  # key dinâmica (remount seguro)
)

st.caption("Dica: você pode digitar fórmulas na célula (ex.: -350, +10%, *1,2, 1000-350). Confirme com Enter.")

# --------------------------------------------------------------------------------------
# Pós-edição: recalcula totais e persiste no estado
#   - Se houve reversão neste run, NÃO sobrescrevemos o state com o dataset do grid
#     (pode estar desatualizado). Em vez disso, usamos o state já revertido.
# --------------------------------------------------------------------------------------
if just_reverted:
    df_edit = st.session_state["volumes_wide"][ano_sel][["Família Comercial"] + month_cols + ["Total Ano"]].copy()
else:
    df_edit = pd.DataFrame(grid_resp["data"]).copy()
    for c in month_cols:
        df_edit[c] = pd.to_numeric(df_edit[c], errors="coerce").fillna(0).round(0).astype(int)
    df_edit["Total Ano"] = df_edit[month_cols].sum(axis=1).astype(int)
    st.session_state.setdefault("volumes_wide", {})
    st.session_state["volumes_wide"][ano_sel] = df_edit[["Família Comercial"] + month_cols + ["Total Ano"]].copy()

# Versão “long” para DRE (sempre do state atual)
long_from_state = st.session_state["volumes_wide"][ano_sel][["Família Comercial"] + month_cols].copy()
long_df = long_from_state.melt(id_vars=["Família Comercial"], var_name="mes_nome", value_name="volume")
rev_map = {v: k for k, v in map_m.items()}
long_df["mes"] = long_df["mes_nome"].map(rev_map).astype(int)
long_df["ano"] = int(ano_sel)
long_df = long_df.drop(columns=["mes_nome"])
long_df["volume"] = pd.to_numeric(long_df["volume"], errors="coerce").fillna(0).round(0).astype(int)
long_df = long_df.sort_values(["Família Comercial", "mes"]).reset_index(drop=True)
st.session_state["volumes_edit"] = long_df

# --------------------------------------------------------------------------------------
# Export CSV da tabela principal (com TOTAL como primeira linha)
# --------------------------------------------------------------------------------------
export_df = st.session_state["volumes_wide"][ano_sel][["Família Comercial"] + month_cols + ["Total Ano"]].copy()
total_row = {"Família Comercial": "TOTAL", **{m: int(export_df[m].sum()) for m in month_cols}, "Total Ano": int(export_df["Total Ano"].sum())}
export_df_out = pd.concat([pd.DataFrame([total_row]), export_df], ignore_index=True)
csv_table_bytes = export_df_out.to_csv(index=False, encoding="utf-8-sig")

with toolbar:
    tc1, tc2, tc3, _ = st.columns([1, 1.8, 2, 5])
    if tc1.button("↩️ Retornar original", key="btn_reset"):
        st.session_state["ask_reset"] = True
    tc2.download_button("💾 Salvar tabela (CSV)", data=csv_table_bytes, file_name=f"tabela_volumes_{ano_sel}.csv", mime="text/csv")
    tc3.info(f"Escala aplicada: {escala_lbl}")

# Confirmador de retorno ao original
if st.session_state.get("ask_reset", False):
    box = st.container()
    with box:
        st.warning("Você quer salvar a simulação atual antes de retornar os valores originais?")
        rc1, rc2, rc3 = st.columns([2, 1, 1])

        rc1.download_button("⬇️ Salvar simulação atual (CSV)", data=csv_table_bytes, file_name=f"simulacao_{ano_sel}.csv", mime="text/csv")
        confirm = rc2.button("✅ Confirmar retorno", key="confirm_reset")
        cancel  = rc3.button("Cancelar", key="cancel_reset")

        if confirm:
            reset_df = orig_wide.copy()
            for c in month_cols:
                reset_df[c] = pd.to_numeric(reset_df[c], errors="coerce").fillna(0).round(0).astype(int)
            reset_df["Total Ano"] = reset_df[month_cols].sum(axis=1).astype(int)
            st.session_state["volumes_wide"][ano_sel] = reset_df[["Família Comercial"] + month_cols + ["Total Ano"]].copy()

            # Remount do grid principal para refletir o “retorno ao original”
            st.session_state["vol_grid_ver"] += 1

            st.session_state["ask_reset"] = False
            st.rerun()
        elif cancel:
            st.session_state["ask_reset"] = False
            st.info("Ação cancelada.")

# Mensagem total do ano
total_ano_all = int(st.session_state["volumes_wide"][ano_sel]["Total Ano"].sum())
st.success(f"Volumes atualizados para {ano_sel}. Total do ano: {total_ano_all:,}".replace(",", "."))

# --------------------------------------------------------------------------------------
# Tabela “De → Para” com exclusão imediata (via fila + key dinâmica)
# --------------------------------------------------------------------------------------
st.markdown("### 🔁 De → Para (alterações vs base do ano selecionado)")

# Alinha famílias/meses para comparar: EDIT = sempre o state atual
edit_wide_current = st.session_state["volumes_wide"][ano_sel][["Família Comercial"] + month_cols].copy()

orig_idx = orig_wide.set_index("Família Comercial")[month_cols]
edit_idx = edit_wide_current.set_index("Família Comercial")[month_cols]
all_fams = orig_idx.index.union(edit_idx.index)
orig_aln = orig_idx.reindex(all_fams).fillna(0).astype(int)
edit_aln = edit_idx.reindex(all_fams).fillna(0).astype(int)

# Long + merge
orig_long = orig_aln.reset_index().melt(id_vars=["Família Comercial"], var_name="Mês", value_name="Antes")
edit_long = edit_aln.reset_index().melt(id_vars=["Família Comercial"], var_name="Mês", value_name="Depois")
mov_df = orig_long.merge(edit_long, on=["Família Comercial", "Mês"], how="outer").fillna(0)

# Filtra apenas onde houve mudança
mov_df["Δ"] = (mov_df["Depois"] - mov_df["Antes"]).astype(int)
mov_df = mov_df.loc[mov_df["Δ"] != 0].copy()
mov_df["Tipo"] = mov_df["Δ"].apply(lambda x: "Inclusão" if x > 0 else "Retirada")

if mov_df.empty:
    st.info("Nenhuma alteração em relação à base do Excel para o ano selecionado.")
else:
    # Colunas finais + gatilho EXCLUIR_AGORA (oculto)
    mov_df = mov_df[["Família Comercial", "Mês", "Tipo", "Antes", "Depois", "Δ"]].copy()
    mov_df["EXCLUIR_AGORA"] = 0
    mov_df["Excluir"] = "✖"  # valor simbólico (estilizado como "botão")

    # Exibição com escala (somente visual)
    if scale_factor == 1:
        mov_vf_body = "return Math.round(val).toLocaleString('pt-BR');"
    else:
        mov_vf_body = "return val.toLocaleString('pt-BR', {minimumFractionDigits:2, maximumFractionDigits:2});"
    mov_value_formatter = JsCode(f"""
    function(params) {{
      if (params.value === null || params.value === undefined) return '';
      const val = Number(params.value) / {scale_factor};
      {mov_vf_body}
    }}
    """)

    # valueGetter para "Excluir" + estilo de botão (sem retornar HTMLElement)
    delete_value_getter = JsCode("function(params){ return '✖'; }")
    delete_cell_style  = JsCode("""
    function(params){
      return {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: '#DC2626',
        color: '#FFFFFF',
        fontWeight: '700',
        cursor: 'pointer',
        borderRadius: '4px'
      };
    }
    """)

    # Click handler: marca EXCLUIR_AGORA=1 (não recalcula aqui!)
    on_cell_clicked = JsCode("""
    function(params) {
      try {
        if (params.colDef && params.colDef.field === 'Excluir') {
          params.node.setDataValue('EXCLUIR_AGORA', 1);
        }
      } catch (e) {}
    }
    """)

    # Config do grid de movimentações
    gbm = GridOptionsBuilder.from_dataframe(mov_df)
    gbm.configure_column("Família Comercial", header_name="Família", editable=False)
    gbm.configure_column("Mês", editable=False)
    gbm.configure_column("Tipo", editable=False)
    gbm.configure_column("Antes", type=["rightAligned","numericColumn"], valueFormatter=mov_value_formatter, editable=False)
    gbm.configure_column("Depois", type=["rightAligned","numericColumn"], valueFormatter=mov_value_formatter, editable=False)
    gbm.configure_column("Δ",     type=["rightAligned","numericColumn"], valueFormatter=mov_value_formatter, editable=False)
    gbm.configure_column("EXCLUIR_AGORA", header_name="", hide=True, editable=False)
    gbm.configure_column("Excluir",
                         header_name="",
                         editable=False,
                         valueGetter=delete_value_getter,
                         cellStyle=delete_cell_style,
                         width=50, max_width=60)

    gbm.configure_grid_options(
        rowHeight=36,
        suppressRowClickSelection=True,
        stopEditingWhenCellsLoseFocus=True,
        onCellClicked=on_cell_clicked
    )
    mov_options = gbm.build()

    # key dinâmica (versão) — evita grid manter estado do clique após rerun
    mov_grid_key = f"mov_grid_v{st.session_state['mov_grid_ver']}"

    grid_resp_mov = AgGrid(
        mov_df,
        gridOptions=mov_options,
        data_return_mode="AS_INPUT",
        update_mode=GridUpdateMode.MODEL_CHANGED,  # reage ao EXCLUIR_AGORA
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,
        theme="balham",
        height=380,
        key=mov_grid_key
    )

    # Captura eventos de exclusão e enfileira reversões (para o PRÓXIMO run)
    mov_after = pd.DataFrame(grid_resp_mov["data"])
    to_del = mov_after.loc[mov_after.get("EXCLUIR_AGORA", 0) == 1].copy()
    if not to_del.empty:
        ops = [{"ano": ano_sel, "fam": r["Família Comercial"], "mes_nome": r["Mês"]} for _, r in to_del.iterrows()]
        st.session_state.setdefault("pending_reverts", []).extend(ops)
        # Bump na versão do grid inferior e rerun — reversões aplicam no topo do próximo run
        st.session_state["mov_grid_ver"] += 1
        st.rerun()

    # Export CSV das alterações (números crus)
    csv_changes = mov_df.drop(columns=["EXCLUIR_AGORA", "Excluir"]).rename(columns={"Família Comercial": "Familia"}).to_csv(index=False, encoding="utf-8-sig")
    st.download_button("⬇️ Baixar alterações (CSV)", data=csv_changes, file_name=f"movimentacoes_{ano_sel}.csv", mime="text/csv")
