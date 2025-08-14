# pages/02_Volumes.py
# --------------------------------------------------------------------------------------
# Módulo: Volumes (Simulador DRE) - Versão Refatorada
#
# Melhorias implementadas:
# - Arquitetura limpa com classes de responsabilidade única
# - Validação robusta de dados com tratamento de erros
# - JavaScript modularizado e testável
# - Sistema de estado simplificado e previsível
# - Performance otimizada com cache
# - Logging para debugging
# - Código maintível e extensível
#
# Arquitetura:
# - VolumeState: Gerencia estado de forma centralizada
# - VolumeValidator: Validação de dados de entrada
# - VolumeFormatter: Formatação e parsing de valores
# - VolumeGrid: Configuração e management dos grids AgGrid
# - VolumeExporter: Funcionalidades de export
# --------------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from core.state import init_state
from core.io import load_volume_base

# AgGrid
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from st_aggrid.shared import JsCode

# --------------------------------------------------------------------------------------
# Setup inicial e logging
# --------------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Volumes", page_icon="🧮", layout="wide")
init_state()


# --------------------------------------------------------------------------------------
# Classes de domínio e utilitárias
# --------------------------------------------------------------------------------------

@dataclass
class VolumeChange:
    """Representa uma alteração de volume"""
    family: str
    month: str
    old_value: int
    new_value: int
    year: int

    @property
    def delta(self) -> int:
        return self.new_value - self.old_value

    @property
    def change_type(self) -> str:
        return "Inclusão" if self.delta > 0 else "Retirada"


class VolumeValidator:
    """Responsável por validar dados de volume"""

    REQUIRED_COLUMNS = ['ano', 'mes', 'volume', 'Família Comercial']
    VALID_MONTHS = list(range(1, 13))

    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame) -> None:
        """Valida estrutura do DataFrame"""
        if df.empty:
            raise ValueError("DataFrame está vazio")

        missing_cols = [col for col in cls.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Colunas obrigatórias ausentes: {missing_cols}")

        # Valida tipos e valores
        if not pd.api.types.is_numeric_dtype(df['ano']):
            raise ValueError("Coluna 'ano' deve ser numérica")

        invalid_months = df['mes'].dropna().unique()
        invalid_months = [m for m in invalid_months if m not in cls.VALID_MONTHS]
        if invalid_months:
            raise ValueError(f"Meses inválidos encontrados: {invalid_months}")

    @classmethod
    def sanitize_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Sanitiza e converte tipos do DataFrame"""
        df = df.copy()

        # Conversões seguras
        df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")
        df["mes"] = pd.to_numeric(df["mes"], errors="coerce").astype("Int64")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(float)

        # Remove linhas com dados críticos faltando
        df = df.dropna(subset=['ano', 'mes', 'Família Comercial'])

        return df


class VolumeFormatter:
    """Responsável por formatação e parsing de valores"""

    MONTH_MAP = {
        1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
        7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"
    }

    SCALE_OPTIONS = {"1x": 1, "1.000x": 1_000, "1.000.000x": 1_000_000}

    @classmethod
    def get_month_columns(cls) -> List[str]:
        """Retorna lista de colunas de meses"""
        return [cls.MONTH_MAP[i] for i in range(1, 13)]

    @classmethod
    def get_scale_factor(cls, scale_label: str) -> int:
        """Retorna fator de escala baseado no label"""
        return cls.SCALE_OPTIONS.get(scale_label, 1)

    @classmethod
    def create_value_formatter_js(cls, scale_factor: int) -> JsCode:
        """Cria formatter JavaScript para valores com escala"""
        if scale_factor == 1:
            format_body = "return Math.round(val).toLocaleString('pt-BR');"
        else:
            format_body = "return val.toLocaleString('pt-BR', {minimumFractionDigits:2, maximumFractionDigits:2});"

        js_code = f"""
        function(params) {{
            if (params.value === null || params.value === undefined) return '';
            const val = Number(params.value) / {scale_factor};
            {format_body}
        }}
        """
        return JsCode(js_code)

    @classmethod
    def create_value_parser_js(cls, scale_factor: int) -> JsCode:
        """Cria parser JavaScript para fórmulas Excel-like"""
        js_code = f"""
        function(params) {{
            var rawCurrent = Number((params.oldValue !== undefined ? params.oldValue : params.value) || 0);
            var input = (params.newValue ?? '').toString().trim();
            var SCALE = {scale_factor};

            if (input === '') return rawCurrent;

            // Função para converter formato BR para float
            function brToFloat(s) {{
                if (s === null || s === undefined) return NaN;
                s = String(s).trim();
                if (s === '') return NaN;
                if (s.indexOf(',') >= 0) {{
                    s = s.replace(/\\./g,'').replace(/,/g,'.');
                }}
                return parseFloat(s);
            }}

            var m;

            // Percentuais: +10% / -7,5%
            m = input.match(/^([+-])\\s*([0-9]+[\\.,]?[0-9]*)\\s*%$/);
            if (m) {{
                var sign = (m[1] === '+') ? 1 : -1;
                var pct = brToFloat(m[2]) / 100.0;
                if (isNaN(pct)) return rawCurrent;
                var newRaw = Math.round(rawCurrent * (1 + sign * pct));
                return Math.max(0, newRaw);
            }}

            // Operações aditivas: +N / -N
            m = input.match(/^([+-])\\s*([0-9]+(?:[\\.,][0-9]+)?)\\s*$/);
            if (m) {{
                var sign = (m[1] === '+') ? 1 : -1;
                var deltaDisp = brToFloat(m[2]);
                if (isNaN(deltaDisp)) return rawCurrent;
                var newRaw = Math.round(rawCurrent + sign * deltaDisp * SCALE);
                return Math.max(0, newRaw);
            }}

            // Multiplicação: *fator
            m = input.match(/^\\*\\s*([0-9]+(?:[\\.,][0-9]+)?)\\s*$/);
            if (m) {{
                var factor = brToFloat(m[1]);
                if (isNaN(factor)) return rawCurrent;
                var newRaw = Math.round(rawCurrent * factor);
                return Math.max(0, newRaw);
            }}

            // Divisão: /divisor  
            m = input.match(/^\\/\\s*([0-9]+(?:[\\.,][0-9]+)?)\\s*$/);
            if (m) {{
                var div = brToFloat(m[1]);
                if (!div) return rawCurrent;
                var newRaw = Math.round(rawCurrent / div);
                return Math.max(0, newRaw);
            }}

            // Expressão absoluta (sanitizada)
            var expr = input;
            if (expr.indexOf(',') >= 0) {{
                expr = expr.replace(/\\./g,'').replace(/,/g,'.');
            }}

            // Permite apenas caracteres seguros
            if (/[^0-9+\\-*/().\\s]/.test(expr)) {{
                var n = brToFloat(input);
                if (!isNaN(n)) {{
                    var newRaw = Math.round(n * SCALE);
                    return Math.max(0, newRaw);
                }}
                return rawCurrent;
            }}

            try {{
                var valDisp = Function('"use strict"; return (' + expr + ')')();
                if (typeof valDisp !== 'number' || !isFinite(valDisp)) return rawCurrent;
                var newRaw = Math.round(valDisp * SCALE);
                return Math.max(0, newRaw);
            }} catch(e) {{
                return rawCurrent;
            }}
        }}
        """
        return JsCode(js_code)


class VolumeState:
    """Gerencia o estado dos volumes de forma centralizada"""

    def __init__(self):
        self._ensure_state_initialized()

    def _ensure_state_initialized(self):
        """Garante que o estado está inicializado"""
        if "volumes_wide" not in st.session_state:
            st.session_state["volumes_wide"] = {}
        if "grid_version" not in st.session_state:
            st.session_state["grid_version"] = 0
        if "changes_log" not in st.session_state:
            st.session_state["changes_log"] = {}

    def has_data_for_year(self, year: int) -> bool:
        """Verifica se há dados para o ano"""
        return year in st.session_state["volumes_wide"]

    def get_data_for_year(self, year: int) -> Optional[pd.DataFrame]:
        """Retorna dados para o ano especificado"""
        return st.session_state["volumes_wide"].get(year)

    def set_data_for_year(self, year: int, data: pd.DataFrame):
        """Define dados para o ano"""
        st.session_state["volumes_wide"][year] = data.copy()
        logger.info(f"Dados atualizados para o ano {year}")

    def reset_to_original(self, year: int, original_data: pd.DataFrame):
        """Reseta dados para valores originais"""
        self.set_data_for_year(year, original_data)
        # Limpa log de alterações
        if year in st.session_state["changes_log"]:
            del st.session_state["changes_log"][year]
        self._increment_grid_version()
        logger.info(f"Dados resetados para valores originais - ano {year}")

    def revert_change(self, year: int, family: str, month: str, original_value: int):
        """Reverte uma alteração específica"""
        if not self.has_data_for_year(year):
            return

        data = self.get_data_for_year(year)
        mask = data["Família Comercial"] == family
        if mask.any():
            data.loc[mask, month] = original_value
            # Recalcula total
            month_cols = VolumeFormatter.get_month_columns()
            data["Total Ano"] = data[month_cols].sum(axis=1).astype(int)

            self.set_data_for_year(year, data)
            self._remove_from_changes_log(year, family, month)
            self._increment_grid_version()
            logger.info(f"Alteração revertida: {family} - {month} = {original_value}")

    def _increment_grid_version(self):
        """Incrementa versão do grid para forçar remount"""
        # Limita crescimento da versão
        st.session_state["grid_version"] = (st.session_state["grid_version"] + 1) % 1000

    def get_grid_version(self) -> int:
        """Retorna versão atual do grid"""
        return st.session_state["grid_version"]

    def _remove_from_changes_log(self, year: int, family: str, month: str):
        """Remove entrada do log de alterações"""
        if year not in st.session_state["changes_log"]:
            return

        key = f"{family}_{month}"
        if key in st.session_state["changes_log"][year]:
            del st.session_state["changes_log"][year][key]


class VolumeGrid:
    """Responsável pela configuração e management dos grids AgGrid"""

    @staticmethod
    def create_row_style_js() -> JsCode:
        """Cria estilo para linhas do grid"""
        return JsCode("""
        function(params) {
            if (params.node.rowPinned === 'top' || 
                (params.data && params.data["Família Comercial"] === "TOTAL")) {
                return {
                    'backgroundColor': '#374151',
                    'color': '#F9FAFB', 
                    'fontWeight': 'bold'
                };
            }
            return null;
        }
        """)

    @staticmethod
    def create_delete_button_js() -> Tuple[JsCode, JsCode, JsCode]:
        """Cria componentes para botão de deletar"""
        value_getter = JsCode("function(params){ return '✖'; }")

        cell_style = JsCode("""
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

        click_handler = JsCode("""
        function(params) {
            try {
                if (params.colDef && params.colDef.field === 'Excluir') {
                    params.node.setDataValue('DELETE_FLAG', 1);
                }
            } catch (e) {
                console.error('Erro no click handler:', e);
            }
        }
        """)

        return value_getter, cell_style, click_handler

    @staticmethod
    def build_main_grid(data: pd.DataFrame, scale_factor: int, grid_version: int) -> GridOptionsBuilder:
        """Constrói grid principal de edição"""
        month_cols = VolumeFormatter.get_month_columns()
        value_formatter = VolumeFormatter.create_value_formatter_js(scale_factor)
        value_parser = VolumeFormatter.create_value_parser_js(scale_factor)
        row_style = VolumeGrid.create_row_style_js()

        # Linha de total
        totals_dict = {
            "Família Comercial": "TOTAL",
            **{col: int(data[col].sum()) for col in month_cols},
            "Total Ano": int(data["Total Ano"].sum())
        }

        gb = GridOptionsBuilder.from_dataframe(data)
        gb.configure_column("Família Comercial", header_name="Família", editable=False)
        gb.configure_column("Total Ano",
                            type=["rightAligned", "numericColumn"],
                            valueFormatter=value_formatter,
                            editable=False)

        for col in month_cols:
            gb.configure_column(col,
                                type=["rightAligned", "numericColumn"],
                                valueFormatter=value_formatter,
                                valueParser=value_parser,
                                editable=True)

        gb.configure_grid_options(
            pinnedTopRowData=[totals_dict],
            getRowStyle=row_style,
            rowHeight=34,
            ensureDomOrder=True,
            enableRangeSelection=True,
            stopEditingWhenCellsLoseFocus=True,
            enterMovesDownAfterEdit=True,
            singleClickEdit=True,
        )

        return gb, totals_dict

    @staticmethod
    def build_changes_grid(changes_df: pd.DataFrame, scale_factor: int) -> GridOptionsBuilder:
        """Constrói grid de alterações com botão de exclusão"""
        value_formatter = VolumeFormatter.create_value_formatter_js(scale_factor)
        del_getter, del_style, del_click = VolumeGrid.create_delete_button_js()

        # Adiciona colunas de controle
        changes_df = changes_df.copy()
        changes_df["DELETE_FLAG"] = 0
        changes_df["Excluir"] = "✖"

        gb = GridOptionsBuilder.from_dataframe(changes_df)
        gb.configure_column("Família Comercial", header_name="Família", editable=False)
        gb.configure_column("Mês", editable=False)
        gb.configure_column("Tipo", editable=False)
        gb.configure_column("Antes",
                            type=["rightAligned", "numericColumn"],
                            valueFormatter=value_formatter,
                            editable=False)
        gb.configure_column("Depois",
                            type=["rightAligned", "numericColumn"],
                            valueFormatter=value_formatter,
                            editable=False)
        gb.configure_column("Δ",
                            type=["rightAligned", "numericColumn"],
                            valueFormatter=value_formatter,
                            editable=False)
        gb.configure_column("DELETE_FLAG", header_name="", hide=True, editable=False)
        gb.configure_column("Excluir",
                            header_name="",
                            editable=False,
                            valueGetter=del_getter,
                            cellStyle=del_style,
                            width=50,
                            max_width=60)

        gb.configure_grid_options(
            rowHeight=36,
            suppressRowClickSelection=True,
            stopEditingWhenCellsLoseFocus=True,
            onCellClicked=del_click
        )

        return gb


class VolumeExporter:
    """Responsável pelas funcionalidades de export"""

    @staticmethod
    def create_table_csv(data: pd.DataFrame, year: int) -> bytes:
        """Cria CSV da tabela principal com linha de total"""
        month_cols = VolumeFormatter.get_month_columns()
        export_df = data[["Família Comercial"] + month_cols + ["Total Ano"]].copy()

        # Adiciona linha de total no topo
        total_row = {
            "Família Comercial": "TOTAL",
            **{col: int(export_df[col].sum()) for col in month_cols},
            "Total Ano": int(export_df["Total Ano"].sum())
        }

        export_df_final = pd.concat([pd.DataFrame([total_row]), export_df], ignore_index=True)
        return export_df_final.to_csv(index=False, encoding="utf-8-sig").encode()

    @staticmethod
    def create_changes_csv(changes_df: pd.DataFrame, year: int) -> bytes:
        """Cria CSV das alterações"""
        export_cols = ["Família Comercial", "Mês", "Tipo", "Antes", "Depois", "Δ"]
        changes_clean = changes_df[export_cols].copy()
        changes_clean = changes_clean.rename(columns={"Família Comercial": "Familia"})
        return changes_clean.to_csv(index=False, encoding="utf-8-sig").encode()


class VolumeProcessor:
    """Classe principal que orquestra todo o processamento"""

    def __init__(self):
        self.state = VolumeState()
        self.validator = VolumeValidator()
        self.formatter = VolumeFormatter()
        self.grid = VolumeGrid()
        self.exporter = VolumeExporter()

    @st.cache_data
    def load_and_process_base_data(_self, data_source) -> pd.DataFrame:
        """Carrega e processa dados base com cache"""
        try:
            if isinstance(data_source, (str, Path)):
                base_df = load_volume_base(data_source)
            else:
                base_df = load_volume_base(data_source)

            _self.validator.validate_dataframe(base_df)
            return _self.validator.sanitize_dataframe(base_df)

        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
            raise

    def create_pivot_from_base(self, base_df: pd.DataFrame, year: int) -> pd.DataFrame:
        """Cria tabela pivot a partir dos dados base"""
        df_year = base_df.loc[base_df["ano"] == year].copy()

        if df_year.empty:
            raise ValueError(f"Não há dados para o ano {year}")

        # Agregação e pivot
        agg_df = df_year.groupby(["Família Comercial", "mes"], as_index=False)["volume"].sum()
        pivot_df = (
            agg_df.pivot(index="Família Comercial", columns="mes", values="volume")
            .reindex(columns=list(range(1, 13)), fill_value=0)
            .fillna(0)
        )

        # Renomeia colunas para nomes dos meses
        pivot_df.columns = [self.formatter.MONTH_MAP[col] for col in pivot_df.columns]
        pivot_df = pivot_df.reset_index()

        # Converte para inteiros e adiciona total
        month_cols = self.formatter.get_month_columns()
        for col in month_cols:
            pivot_df[col] = pd.to_numeric(pivot_df[col], errors="coerce").fillna(0).round(0).astype(int)

        pivot_df["Total Ano"] = pivot_df[month_cols].sum(axis=1).astype(int)

        return pivot_df

    def calculate_changes(self, original_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
        """Calcula alterações entre dados originais e atuais"""
        month_cols = self.formatter.get_month_columns()

        # Prepara dados para comparação
        orig_idx = original_df.set_index("Família Comercial")[month_cols]
        curr_idx = current_df.set_index("Família Comercial")[month_cols]

        all_families = orig_idx.index.union(curr_idx.index)
        orig_aligned = orig_idx.reindex(all_families).fillna(0).astype(int)
        curr_aligned = curr_idx.reindex(all_families).fillna(0).astype(int)

        # Converte para formato long e merge
        orig_long = orig_aligned.reset_index().melt(
            id_vars=["Família Comercial"],
            var_name="Mês",
            value_name="Antes"
        )
        curr_long = curr_aligned.reset_index().melt(
            id_vars=["Família Comercial"],
            var_name="Mês",
            value_name="Depois"
        )

        changes_df = orig_long.merge(curr_long, on=["Família Comercial", "Mês"], how="outer").fillna(0)
        changes_df["Δ"] = (changes_df["Depois"] - changes_df["Antes"]).astype(int)
        changes_df = changes_df.loc[changes_df["Δ"] != 0].copy()
        changes_df["Tipo"] = changes_df["Δ"].apply(lambda x: "Inclusão" if x > 0 else "Retirada")

        return changes_df[["Família Comercial", "Mês", "Tipo", "Antes", "Depois", "Δ"]].copy()


# --------------------------------------------------------------------------------------
# Interface principal
# --------------------------------------------------------------------------------------

def main():
    """Função principal da aplicação"""
    processor = VolumeProcessor()

    st.header("🧮 Volumes por Família — meses em colunas + TOTAL no topo")

    # Carregamento de dados com tratamento robusto de erros
    try:
        data_path = Path("data/base_final_longo_para_modelo.xlsx")

        if not data_path.exists():
            st.warning("Base de volumes não encontrada. Faça o upload abaixo.")
            uploaded_file = st.file_uploader(
                "Enviar base de volumes (.xlsx)",
                type=["xlsx"],
                accept_multiple_files=False,
                key="upload_volumes"
            )

            if uploaded_file is None:
                st.info("Coloque o arquivo na pasta 'data/' com o nome 'base_final_longo_para_modelo.xlsx'")
                st.stop()

            base_df = processor.load_and_process_base_data(uploaded_file)
        else:
            base_df = processor.load_and_process_base_data(data_path)

    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        st.stop()

    # Extrai anos disponíveis
    anos_disponiveis = sorted([int(ano) for ano in base_df["ano"].dropna().unique()])
    if not anos_disponiveis:
        st.error("Nenhum ano válido encontrado nos dados.")
        st.stop()

    # Interface de filtros
    col1, col2, _ = st.columns([1, 1.2, 5])

    with col1:
        ano_selecionado = st.selectbox(
            "Ano",
            options=anos_disponiveis,
            index=len(anos_disponiveis) - 1,
            help="Selecione o ano para editar os volumes"
        )

    with col2:
        escala_label = st.selectbox(
            "Escala de exibição",
            options=["1x", "1.000x", "1.000.000x"],
            index=0
        )

    scale_factor = processor.formatter.get_scale_factor(escala_label)

    # Processa dados para o ano selecionado
    try:
        original_pivot = processor.create_pivot_from_base(base_df, ano_selecionado)
    except ValueError as e:
        st.warning(str(e))
        st.stop()

    # Gerencia estado dos dados editáveis
    if not processor.state.has_data_for_year(ano_selecionado):
        processor.state.set_data_for_year(ano_selecionado, original_pivot)

    current_data = processor.state.get_data_for_year(ano_selecionado)

    # Toolbar com ações
    toolbar = st.container()

    # Grid principal de edição
    try:
        gb_main, totals_dict = processor.grid.build_main_grid(
            current_data,
            scale_factor,
            processor.state.get_grid_version()
        )

        grid_response = AgGrid(
            current_data,
            gridOptions=gb_main.build(),
            data_return_mode="AS_INPUT",
            update_mode=GridUpdateMode.MODEL_CHANGED,
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True,
            theme="balham",
            height=520,
            key=f"main_grid_v{processor.state.get_grid_version()}"
        )

        st.caption("💡 Dica: digite fórmulas na célula (ex.: -350, +10%, *1,2, 1000-350). Confirme com Enter.")

    except Exception as e:
        st.error(f"Erro ao renderizar grid principal: {str(e)}")
        logger.error(f"Erro no grid principal: {e}")
        st.stop()

    # Processa dados editados
    try:
        edited_data = pd.DataFrame(grid_response["data"]).copy()
        month_cols = processor.formatter.get_month_columns()

        # Sanitiza dados editados
        for col in month_cols:
            edited_data[col] = pd.to_numeric(edited_data[col], errors="coerce").fillna(0).round(0).astype(int)

        edited_data["Total Ano"] = edited_data[month_cols].sum(axis=1).astype(int)
        processor.state.set_data_for_year(ano_selecionado, edited_data)

        # Cria versão long para DRE
        long_df = edited_data[["Família Comercial"] + month_cols].melt(
            id_vars=["Família Comercial"],
            var_name="mes_nome",
            value_name="volume"
        )
        month_reverse_map = {v: k for k, v in processor.formatter.MONTH_MAP.items()}
        long_df["mes"] = long_df["mes_nome"].map(month_reverse_map).astype(int)
        long_df["ano"] = ano_selecionado
        long_df = long_df.drop(columns=["mes_nome"]).sort_values(["Família Comercial", "mes"])

        st.session_state["volumes_edit"] = long_df.reset_index(drop=True)

    except Exception as e:
        st.error(f"Erro ao processar dados editados: {str(e)}")
        logger.error(f"Erro processamento dados editados: {e}")

    # Calcula dados para export
    try:
        current_state_data = processor.state.get_data_for_year(ano_selecionado)
        csv_table_bytes = processor.exporter.create_table_csv(current_state_data, ano_selecionado)

    except Exception as e:
        st.error(f"Erro ao preparar export: {str(e)}")
        logger.error(f"Erro no export: {e}")
        csv_table_bytes = b""

    # Toolbar com botões de ação
    with toolbar:
        col1, col2, col3, _ = st.columns([1, 1.8, 2, 5])

        if col1.button("↩️ Retornar original", key="btn_reset"):
            st.session_state["ask_reset"] = True

        col2.download_button(
            "💾 Salvar tabela (CSV)",
            data=csv_table_bytes,
            file_name=f"tabela_volumes_{ano_selecionado}.csv",
            mime="text/csv",
            disabled=len(csv_table_bytes) == 0
        )

        col3.info(f"Escala aplicada: {escala_label}")

    # Modal de confirmação para reset
    if st.session_state.get("ask_reset", False):
        with st.container():
            st.warning("⚠️ Você quer salvar a simulação atual antes de retornar aos valores originais?")

            col1, col2, col3 = st.columns([2, 1, 1])

            col1.download_button(
                "⬇️ Salvar simulação atual (CSV)",
                data=csv_table_bytes,
                file_name=f"simulacao_{ano_selecionado}.csv",
                mime="text/csv",
                disabled=len(csv_table_bytes) == 0
            )

            if col2.button("✅ Confirmar retorno", key="confirm_reset"):
                try:
                    processor.state.reset_to_original(ano_selecionado, original_pivot)
                    st.session_state["ask_reset"] = False
                    st.success("✅ Dados retornados aos valores originais!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro ao resetar dados: {str(e)}")
                    logger.error(f"Erro no reset: {e}")

            if col3.button("❌ Cancelar", key="cancel_reset"):
                st.session_state["ask_reset"] = False
                st.info("Ação cancelada.")

    # Exibe total do ano
    try:
        total_ano = int(current_state_data["Total Ano"].sum())
        st.success(f"📊 Volumes atualizados para {ano_selecionado}. Total do ano: {total_ano:,}".replace(",", "."))
    except Exception as e:
        logger.error(f"Erro ao calcular total do ano: {e}")
        st.warning("Erro ao calcular total do ano")

    # --------------------------------------------------------------------------------------
    # Tabela "De → Para" com alterações
    # --------------------------------------------------------------------------------------
    st.markdown("### 🔁 De → Para (alterações vs base do ano selecionado)")

    try:
        changes_df = processor.calculate_changes(original_pivot, current_state_data)

        if changes_df.empty:
            st.info("ℹ️ Nenhuma alteração em relação à base do Excel para o ano selecionado.")
        else:
            # Configura grid de alterações
            gb_changes = processor.grid.build_changes_grid(changes_df, scale_factor)

            changes_response = AgGrid(
                changes_df,
                gridOptions=gb_changes.build(),
                data_return_mode="AS_INPUT",
                update_mode=GridUpdateMode.MODEL_CHANGED,
                fit_columns_on_grid_load=True,
                allow_unsafe_jscode=True,
                theme="balham",
                height=380,
                key=f"changes_grid_v{processor.state.get_grid_version()}"
            )

            # Processa exclusões de alterações
            try:
                changes_after = pd.DataFrame(changes_response["data"])
                deletions = changes_after.loc[changes_after.get("DELETE_FLAG", 0) == 1]

                if not deletions.empty:
                    for _, row in deletions.iterrows():
                        family = row["Família Comercial"]
                        month = row["Mês"]
                        original_value = int(row["Antes"])

                        processor.state.revert_change(ano_selecionado, family, month, original_value)

                    st.success(f"✅ {len(deletions)} alteração(ões) revertida(s)!")
                    st.rerun()

            except Exception as e:
                logger.error(f"Erro ao processar exclusões: {e}")
                st.error("Erro ao processar exclusões de alterações")

            # Export das alterações
            try:
                csv_changes = processor.exporter.create_changes_csv(changes_df, ano_selecionado)
                st.download_button(
                    "⬇️ Baixar alterações (CSV)",
                    data=csv_changes,
                    file_name=f"movimentacoes_{ano_selecionado}.csv",
                    mime="text/csv"
                )
            except Exception as e:
                logger.error(f"Erro ao exportar alterações: {e}")
                st.warning("Erro ao preparar export das alterações")

    except Exception as e:
        st.error(f"Erro ao calcular alterações: {str(e)}")
        logger.error(f"Erro no cálculo de alterações: {e}")


# --------------------------------------------------------------------------------------
# Ponto de entrada
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Erro crítico na aplicação: {str(e)}")
        logger.critical(f"Erro crítico: {e}")

        # Em desenvolvimento, mostra o traceback completo
        if st.secrets.get("DEBUG", False):
            st.exception(e)