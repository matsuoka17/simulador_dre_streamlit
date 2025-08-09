import streamlit as st
import pandas as pd
from pathlib import Path
from core.state import init_state
from core.io import load_volume_base

# AgGrid (separador de milhar, linha TOTAL fixa, estilo e edição)
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from st_aggrid.shared import JsCode

st.set_page_config(page_title="Volumes", page_icon="🧮", layout="wide")
init_state()

st.header("🧮 Simulador de Volume (Família Comercial)")

# === Toolbar (fica logo abaixo do título; será preenchida depois de calcular a tabela) ===
toolbar = st.container()

# === Carregamento da base ===
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

# Sanitização mínima
base_df["ano"] = pd.to_numeric(base_df["ano"], errors="coerce").astype("Int64")
base_df["mes"] = pd.to_numeric(base_df["mes"], errors="coerce").astype("Int64")
base_df["volume"] = pd.to_numeric(base_df["volume"], errors="coerce").fillna(0).astype(float)

anos = sorted([int(a) for a in base_df["ano"].dropna().unique().tolist()])
if not anos:
    st.error("A coluna 'ano' não possui valores válidos.")
    st.stop()

# === Filtro de Ano ===
colf1, _ = st.columns([1, 4])
with colf1:
    ano_sel = st.selectbox("Ano", options=anos, index=len(anos) - 1, help="Selecione o ano para editar os volumes.")

df_y = base_df.loc[base_df["ano"] == ano_sel].copy()
if df_y.empty:
    st.warning(f"Não há dados para o ano {ano_sel}.")
    st.stop()

# === Agregação e Pivot (Família nas linhas; Meses nas colunas) ===
agg = df_y.groupby(["Família Comercial", "mes"], as_index=False)["volume"].sum()

meses_num = list(range(1, 13))
map_m = {1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun", 7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"}
month_cols = [map_m[m] for m in meses_num]

pivot_base = (
    agg.pivot(index="Família Comercial", columns="mes", values="volume")
       .reindex(columns=meses_num, fill_value=0)
       .fillna(0)
)
pivot_base.columns = [map_m.get(int(c), str(c)) for c in pivot_base.columns]
pivot_base = pivot_base.reset_index()

# >>> Guarda a BASE ORIGINAL (do Excel) para o "Retornar original"
orig_wide = pivot_base[["Família Comercial"] + month_cols].copy()
for c in month_cols:
    orig_wide[c] = pd.to_numeric(orig_wide[c], errors="coerce").fillna(0).round(0).astype(int)
orig_wide["Total Ano"] = orig_wide[month_cols].sum(axis=1).astype(int)

# Recupera edição prévia (só para preencher a grade)
if "volumes_wide" not in st.session_state:
    st.session_state["volumes_wide"] = {}
if ano_sel in st.session_state["volumes_wide"]:
    df_prev = st.session_state["volumes_wide"][ano_sel]
    if set(df_prev.columns) == set(["Família Comercial"] + month_cols + ["Total Ano"]):
        pivot_base = df_prev[["Família Comercial"] + month_cols].copy()

# Inteiros
for c in month_cols:
    if c in pivot_base.columns:
        pivot_base[c] = pd.to_numeric(pivot_base[c], errors="coerce").fillna(0).round(0).astype(int)

pivot_base["Total Ano"] = pivot_base[month_cols].sum(axis=1).astype(int)

# === Linha TOTAL (pinned top) ===
totals_dict = {"Família Comercial": "TOTAL"}
for c in month_cols:
    totals_dict[c] = int(pivot_base[c].sum())
totals_dict["Total Ano"] = int(pivot_base["Total Ano"].sum())

# --- JS: formatador milhar + parser inteiro ---
value_formatter = JsCode("""
function(params) {
  if (params.value === null || params.value === undefined) return '';
  return Number(params.value).toLocaleString('pt-BR');
}
""")
value_parser_int = JsCode("""
function(params) {
  if (params.newValue === null || params.newValue === undefined || params.newValue === '') return 0;
  let s = String(params.newValue).replace(/\\./g,'').replace(',', '.');
  let n = parseFloat(s);
  if (isNaN(n)) return 0;
  return Math.round(n);
}
""")
get_row_style = JsCode("""
function(params) {
  if (params.node.rowPinned === 'top' || (params.data && params.data["Família Comercial"] === "TOTAL")) {
    return {
      'backgroundColor': '#374151',
      'color': '#F9FAFB',
      'fontWeight': 'bold'
    };
  }
  return null;
}
""")

# === Grid Options ===
gb = GridOptionsBuilder.from_dataframe(pivot_base)
gb.configure_column("Família Comercial", editable=False)
gb.configure_column("Total Ano", type=["rightAligned", "numericColumn"], valueFormatter=value_formatter, editable=False)
for col in month_cols:
    gb.configure_column(col, type=["rightAligned", "numericColumn"], valueFormatter=value_formatter, valueParser=value_parser_int, editable=True)

gb.configure_grid_options(
    pinnedTopRowData=[totals_dict],
    getRowStyle=get_row_style,
    rowHeight=34,
    ensureDomOrder=True,
    enableRangeSelection=True,
)

grid_options = gb.build()

grid_resp = AgGrid(
    pivot_base,
    gridOptions=grid_options,
    data_return_mode="AS_INPUT",
    update_mode=GridUpdateMode.VALUE_CHANGED,
    fit_columns_on_grid_load=True,
    allow_unsafe_jscode=True,
    theme="balham",
    height=520,
)

# === Pós-edição: recalcula totais e salva sessão ===
df_edit = pd.DataFrame(grid_resp["data"]).copy()
for c in month_cols:
    df_edit[c] = pd.to_numeric(df_edit[c], errors="coerce").fillna(0).round(0).astype(int)
df_edit["Total Ano"] = df_edit[month_cols].sum(axis=1).astype(int)

# Guarda visão wide do ano (para reabrir com os mesmos valores)
st.session_state["volumes_wide"][ano_sel] = df_edit[["Família Comercial"] + month_cols + ["Total Ano"]].copy()

# Converte para formato longo (para a DRE)
long_df = df_edit.melt(id_vars=["Família Comercial", "Total Ano"], var_name="mes_nome", value_name="volume")
rev_map = {v: k for k, v in map_m.items()}
long_df["mes"] = long_df["mes_nome"].map(rev_map).astype(int)
long_df["ano"] = int(ano_sel)
long_df = long_df.drop(columns=["mes_nome", "Total Ano"])
long_df["volume"] = pd.to_numeric(long_df["volume"], errors="coerce").fillna(0).round(0).astype(int)
long_df = long_df.sort_values(["Família Comercial", "mes"]).reset_index(drop=True)
st.session_state["volumes_edit"] = long_df

# === CSV da TABELA PRINCIPAL (com TOTAL como primeira linha) ===
export_df = df_edit[["Família Comercial"] + month_cols + ["Total Ano"]].copy()
total_row = {"Família Comercial": "TOTAL", **{m: int(export_df[m].sum()) for m in month_cols}, "Total Ano": int(export_df["Total Ano"].sum())}
export_df_out = pd.concat([pd.DataFrame([total_row]), export_df], ignore_index=True)
csv_table_bytes = export_df_out.to_csv(index=False, encoding="utf-8-sig")

# === Toolbar preenchida agora: Retornar Original + Salvar CSV da tabela ===
with toolbar:
    tc1, tc2, _ = st.columns([1, 2, 6])
    if tc1.button("↩️ Retornar original", key="btn_reset"):
        st.session_state["ask_reset"] = True
    tc2.download_button(
        "💾 Salvar tabela (CSV)",
        data=csv_table_bytes,
        file_name=f"tabela_volumes_{ano_sel}.csv",
        mime="text/csv",
        key="btn_save_table_csv"
    )

# === Confirmação de retorno ao original ===
if st.session_state.get("ask_reset", False):
    box = st.container()
    with box:
        st.warning("Você quer salvar a simulação atual antes de retornar os valores originais?")
        rc1, rc2, rc3 = st.columns([2, 1, 1])

        # Oferece salvar a simulação atual (tabela principal)
        rc1.download_button(
            "⬇️ Salvar simulação atual (CSV)",
            data=csv_table_bytes,
            file_name=f"simulacao_{ano_sel}.csv",
            mime="text/csv",
            key="btn_save_snapshot"
        )

        confirm = rc2.button("✅ Confirmar retorno", key="confirm_reset")
        cancel = rc3.button("Cancelar", key="cancel_reset")

        if confirm:
            reset_df = orig_wide.copy()
            for c in month_cols:
                reset_df[c] = pd.to_numeric(reset_df[c], errors="coerce").fillna(0).round(0).astype(int)
            reset_df["Total Ano"] = reset_df[month_cols].sum(axis=1).astype(int)
            st.session_state["volumes_wide"][ano_sel] = reset_df[["Família Comercial"] + month_cols + ["Total Ano"]].copy()
            st.session_state["ask_reset"] = False
            st.success("Tabela retornada aos valores originais.")
            st.experimental_rerun()
        elif cancel:
            st.session_state["ask_reset"] = False
            st.info("Ação cancelada.")

# === Mensagem e totais ===
total_ano_all = int(df_edit["Total Ano"].sum())
st.success(f"Volumes atualizados para {ano_sel}. Total do ano: {total_ano_all:,}".replace(",", "."))

# ===============================
# De → Para UNIFICADO (uma tabela)
# ===============================
st.markdown("### 🔁 De → Para (alterações vs base do ano selecionado)")

# Alinha famílias/meses entre original e editado
orig_idx = orig_wide.set_index("Família Comercial")[month_cols]
edit_idx = df_edit.set_index("Família Comercial")[month_cols]
all_fams = orig_idx.index.union(edit_idx.index)
orig_aln = orig_idx.reindex(all_fams).fillna(0).astype(int)
edit_aln = edit_idx.reindex(all_fams).fillna(0).astype(int)

# Derrete pra comparar célula a célula
orig_long = orig_aln.reset_index().melt(id_vars=["Família Comercial"], var_name="Mês", value_name="Antes")
edit_long = edit_aln.reset_index().melt(id_vars=["Família Comercial"], var_name="Mês", value_name="Depois")
mov_df = orig_long.merge(edit_long, on=["Família Comercial", "Mês"], how="outer").fillna(0)

# Delta e Tipo
mov_df["Δ"] = (mov_df["Depois"] - mov_df["Antes"]).astype(int)
mov_df = mov_df.loc[mov_df["Δ"] != 0].copy()
mov_df["Tipo"] = mov_df["Δ"].apply(lambda x: "Inclusão" if x > 0 else "Retirada")

# Ordena por impacto absoluto desc
mov_df["absΔ"] = mov_df["Δ"].abs()
mov_df = mov_df.sort_values(["absΔ", "Família Comercial", "Mês"], ascending=[False, True, True]).drop(columns=["absΔ"])

# Tabela para exibição (formatada com milhar)
fmt = lambda v: f"{int(v):,}".replace(",", ".")
if not mov_df.empty:
    show_df = mov_df.copy()
    show_df["Antes"] = show_df["Antes"].map(fmt)
    show_df["Depois"] = show_df["Depois"].map(fmt)
    show_df["Δ"] = show_df["Δ"].map(fmt)

    st.dataframe(
        show_df[["Família Comercial", "Mês", "Tipo", "Antes", "Depois", "Δ"]]
            .rename(columns={"Família Comercial": "Família"}),
        use_container_width=True,
        hide_index=True
    )

    # CSV apenas com as alterações (números crus)
    csv_changes = mov_df[["Família Comercial", "Mês", "Tipo", "Antes", "Depois", "Δ"]] \
        .rename(columns={"Família Comercial": "Familia"}) \
        .to_csv(index=False, encoding="utf-8-sig")

    st.download_button(
        "⬇️ Baixar alterações (CSV)",
        data=csv_changes,
        file_name=f"movimentacoes_{ano_sel}.csv",
        mime="text/csv",
        key="btn_save_changes_csv"
    )
else:
    st.info("Nenhuma alteração em relação à base do Excel para o ano selecionado.")
