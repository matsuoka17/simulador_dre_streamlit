# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------------
# Validação — Tecnologia × Família × Volume  (YTG only)
# --------------------------------------------------------------------------------------
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ====== MODELOS DO PROJETO ============================================================
# (dependemos deles só para CURRENT/RES e o mapa de meses)
try:
    from core.models import (
        load_current_long,
        res_volume_by_family_long,
        MONTH_MAP_PT,
    )
except Exception:
    from simulador_dre_streamlit.core.models import (  # type: ignore
        load_current_long,
        res_volume_by_family_long,
        MONTH_MAP_PT,
    )

# --------------------------------------------------------------------------------------
# CONFIGURAÇÃO DE PÁGINA (única)
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Validação — Capacidade x Necessidade (YTG)", page_icon="🧪", layout="wide")

# ======================================================================================
# UTILITÁRIOS
# ======================================================================================

MONTHS_PT = MONTH_MAP_PT  # {1:'Jan', ..., 12:'Dez'}

LOGS: List[str] = []
def log(msg: str):
    LOGS.append(str(msg))

def _first_existing(paths: List[str]) -> Optional[Path]:
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None

def _fmt_int(v: float | int | None, scale: int) -> str:
    if v is None or pd.isna(v):
        return "-"
    vv = float(v) / (scale if scale else 1)
    try:
        return f"{vv:,.0f}".replace(",", ".")
    except Exception:
        return str(v)

def _fmt_signed(v: float | int | None, scale: int) -> str:
    if v is None or pd.isna(v):
        return "-"
    vv = float(v) / (scale if scale else 1)
    return f"{vv:,.0f}".replace(",", ".")

def _year_cutoff_from_current(year: int) -> int:
    """Último mês com volume >0 do cenário Realizado no CURRENT."""
    try:
        cur = load_current_long()
        cur["ano"] = pd.to_numeric(cur["ano"], errors="coerce").astype("Int64")
        cur["mes"] = pd.to_numeric(cur["mes"], errors="coerce").astype("Int64")
        cur["indicador_id"] = cur["indicador_id"].astype(str)
        cur["cenario"] = cur["cenario"].astype(str)

        mask = (
            (cur["ano"] == year)
            & (cur["cenario"].str.contains("Realiz", case=False, na=False))
            & (cur["indicador_id"] == "volume_uc")
        )
        m = cur.loc[mask].groupby("mes")["valor"].sum(min_count=1)
        if m.empty:
            return 0
        nz = m[m > 0]
        return int(nz.index.max()) if not nz.empty else 0
    except Exception as e:
        log(f"⚠️ Não foi possível inferir cutoff do CURRENT: {e}")
        return 0

def _tech_map() -> pd.DataFrame:
    """Lê tecnologias_familia.xlsx; devolve DataFrame com colunas 'Família Comercial','Tecnologia'."""
    p = _first_existing([
        "data/tec_poh/tecnologias_familia.xlsx",
        "data/tecnologias_familia.xlsx",
    ])
    if not p:
        log("⚠️ tecnologias_familia.xlsx não encontrado; toda família cairá na tecnologia 'N/D'.")
        return pd.DataFrame(columns=["Família Comercial", "Tecnologia"])
    try:
        df = pd.read_excel(p)
        cols = {c.lower(): c for c in df.columns}
        fam = cols.get("família comercial") or cols.get("familia comercial") or cols.get("familia")
        tec = cols.get("tecnologia") or cols.get("tec")
        if not fam or not tec:
            raise ValueError("Planilha de tecnologias deve ter colunas 'Família Comercial' e 'Tecnologia'.")
        out = df[[fam, tec]].rename(columns={fam: "Família Comercial", tec: "Tecnologia"}).dropna()
        out = out.drop_duplicates(subset=["Família Comercial"], keep="first")
        return out
    except Exception as e:
        log(f"⚠️ Falha ao ler tecnologias_familia.xlsx: {e}")
        return pd.DataFrame(columns=["Família Comercial", "Tecnologia"])

def _capacidade_por_tec(year: int, months: List[int]) -> pd.DataFrame:
    """
    Lê capacidade_tecnologias.xlsx:
      - Aceita uma coluna única 'Capacidade' → replica para todos os meses (YTG)
      - Ou colunas por mês (Jan..Dez)
    Retorna DF com colunas ['Tecnologia'] + meses pt-br do YTG.
    """
    p = _first_existing([
        "data/cap_poh/capacidade_tecnologias.xlsx",
        "data/capacidade_tecnologias.xlsx",
    ])
    if not p:
        log("⚠️ capacidade_tecnologias.xlsx não encontrado; capacidade=0.")
        return pd.DataFrame(columns=["Tecnologia"] + [MONTHS_PT[m] for m in months])

    raw = pd.read_excel(p)
    cols = {c.lower(): c for c in raw.columns}
    tec = cols.get("tecnologia") or cols.get("tec")
    if not tec:
        log("⚠️ capacidade_tecnologias.xlsx sem coluna 'Tecnologia'.")
        return pd.DataFrame(columns=["Tecnologia"] + [MONTHS_PT[m] for m in months])

    # Detecta se há colunas por mês
    mes_cols = [cols.get(MONTHS_PT[m].lower()) for m in months]
    if all(c is not None for c in mes_cols):
        df = raw[[tec, *mes_cols]].rename(columns={tec: "Tecnologia", **{c: MONTHS_PT[m] for c, m in zip(mes_cols, months)}})
        for mc in [MONTHS_PT[m] for m in months]:
            df[mc] = pd.to_numeric(df[mc], errors="coerce").fillna(0).astype(float)
        return df
    # Ou uma única coluna 'Capacidade'
    cap = cols.get("capacidade")
    if cap:
        df = raw[[tec, cap]].rename(columns={tec: "Tecnologia", cap: "Capacidade"})
        for mc in [MONTHS_PT[m] for m in months]:
            df[mc] = pd.to_numeric(df["Capacidade"], errors="coerce").fillna(0).astype(float)
        return df.drop(columns=["Capacidade"])
    log("⚠️ capacidade_tecnologias.xlsx sem colunas mensais nem 'Capacidade'. Capacidade=0.")
    out = raw[[tec]].rename(columns={tec: "Tecnologia"}).copy()
    for mc in [MONTHS_PT[m] for m in months]:
        out[mc] = 0.0
    return out

def _estoque_inicial_por_familia(year: int) -> pd.DataFrame:
    """
    Lê estoque_inicial.xlsx. Aceita colunas:
      - 'Família Comercial' + 'Estoque Inicial' (para o ano todo)
      - ou com 'Ano' e/ou 'Mês' (usaremos o que existir)
    """
    p = _first_existing([
        "data/estoque/estoque_inicial.xlsx",
        "data/estoque_inicial.xlsx",
    ])
    if not p:
        log("ℹ️ estoque_inicial.xlsx não encontrado; estoque inicial = 0.")
        return pd.DataFrame(columns=["Família Comercial", "Mês", "Estoque Inicial"])

    df = pd.read_excel(p)
    cols = {c.lower(): c for c in df.columns}
    fam = cols.get("família comercial") or cols.get("familia comercial") or cols.get("familia")
    if not fam:
        log("⚠️ estoque_inicial.xlsx sem coluna 'Família Comercial'. Estoque inicial=0.")
        return pd.DataFrame(columns=["Família Comercial", "Mês", "Estoque Inicial"])

    col_ei = cols.get("estoque inicial") or cols.get("estoque") or cols.get("ei")
    if not col_ei:
        log("⚠️ estoque_inicial.xlsx sem coluna 'Estoque Inicial'. Estoque inicial=0.")
        return pd.DataFrame(columns=["Família Comercial", "Mês", "Estoque Inicial"])

    ano_col = cols.get("ano") or cols.get("ano (yyyy)") or cols.get("year")
    mes_col = cols.get("mês") or cols.get("mes") or cols.get("month")

    out = df[[fam, col_ei]].rename(columns={fam: "Família Comercial", col_ei: "Estoque Inicial"}).copy()
    out["Estoque Inicial"] = pd.to_numeric(out["Estoque Inicial"], errors="coerce").fillna(0).astype(float)

    if ano_col:
        out["Ano"] = pd.to_numeric(df[ano_col], errors="coerce").astype("Int64")
    else:
        out["Ano"] = year

    if mes_col:
        out["Mês"] = pd.to_numeric(df[mes_col], errors="coerce").astype("Int64")
    else:
        out["Mês"] = pd.NA  # se não tiver mês, vamos aplicar só no 1º YTG

    return out

def _sales_source_long(year: int, use_ui: bool) -> pd.DataFrame:
    """
    Retorna long: ['Família Comercial','ano','mes','volume']
    Fonte = UI (toggle) OU RES_WORKING.
    """
    if use_ui:
        # da UI: st.session_state['volumes_wide'][year] (Família x meses)
        vw = st.session_state.get("volumes_wide")
        if isinstance(vw, dict) and year in vw and isinstance(vw[year], pd.DataFrame):
            w = vw[year].copy()
            fam_col = None
            for c in w.columns:
                if str(c).lower().startswith("fam"):
                    fam_col = c
                    break
            if fam_col is None:
                fam_col = w.columns[0]
            long = []
            for m in range(1, 13):
                mc = MONTHS_PT[m]
                if mc in w.columns:
                    tmp = pd.DataFrame({
                        "Família Comercial": w[fam_col].astype(str),
                        "ano": year,
                        "mes": m,
                        "volume": pd.to_numeric(w[mc], errors="coerce").fillna(0).astype(float),
                    })
                    long.append(tmp)
            return pd.concat(long, ignore_index=True) if long else pd.DataFrame(columns=["Família Comercial","ano","mes","volume"])
        log("ℹ️ volumes_wide (UI) não disponível; usando RES_WORKING.")
    try:
        df = res_volume_by_family_long()
        if df is None or df.empty:
            return pd.DataFrame(columns=["Família Comercial","ano","mes","volume"])
        df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")
        df["mes"] = pd.to_numeric(df["mes"], errors="coerce").astype("Int64")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(float)
        df["Família Comercial"] = df["Família Comercial"].astype(str)
        return df[df["ano"] == year].copy()
    except Exception as e:
        log(f"⚠️ Falha ao ler RES_WORKING: {e}")
        return pd.DataFrame(columns=["Família Comercial","ano","mes","volume"])

def _ytg_months(year: int) -> List[int]:
    cutoff = _year_cutoff_from_current(year)
    return [m for m in range(cutoff + 1, 13)]

def _apply_estoque_inicial(first_m: int, families: List[str], ei_df: pd.DataFrame) -> Dict[str, float]:
    """
    Retorna dict {fam: estoque_inicial_aplicado_no_primeiro_mes_YTG}
    Regras:
     - Se planilha tem Mês → usa o valor daquele mês (= first_m) senão, usa o valor “geral”
     - Se houver múltiplas linhas para a família, soma.
    """
    out: Dict[str, float] = {}
    if ei_df is None or ei_df.empty:
        return {f: 0.0 for f in families}

    same_month = ei_df.loc[ei_df["Mês"].fillna(first_m).astype(int) == int(first_m)]
    for fam in families:
        v = pd.to_numeric(same_month.loc[same_month["Família Comercial"] == fam, "Estoque Inicial"], errors="coerce").fillna(0).sum()
        out[fam] = float(v)
    return out

def _style_gap(val: float) -> str:
    if pd.isna(val):
        return ""
    if val < 0:
        return "background-color:#fde2e2; color:#991b1b; font-weight:600"
    if val > 0:
        return "background-color:#ecfdf5; color:#065f46"
    return ""

# ======================================================================================
# CÁLCULO PRINCIPAL (YTG)
# ======================================================================================

def compute_by_technology(year: int, use_ui: bool) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Retorna:
      - kpis[tec]: Tabela KPI por tecnologia (linhas: Vendas, Estoque Inicial, Necessidade Produção, Capacidade, Gap)
      - detail[tec]: Detalhamento por família (linhas famílias/Total, colunas meses YTG)
      - resumo_tec: Capacidade vs Necessidade somadas (por tecnologia)
    """
    months = _ytg_months(year)
    if not months:
        return {}, {}, pd.DataFrame()

    months_pt = [MONTHS_PT[m] for m in months]

    # VENDAS long → pivot (Família x meses YTG)
    sales = _sales_source_long(year, use_ui)
    if sales.empty:
        log("⚠️ Fonte de vendas (UI/RES) vazia para o ano.")
    for c in ("ano","mes"):
        if c in sales.columns:
            sales[c] = pd.to_numeric(sales[c], errors="coerce").astype("Int64")
    sales["volume"] = pd.to_numeric(sales.get("volume", 0), errors="coerce").fillna(0).astype(float)
    sales["Família Comercial"] = sales.get("Família Comercial", "").astype(str)
    sales = sales[sales["mes"].isin(months)].copy()

    piv_sales = (sales
                 .groupby(["Família Comercial","mes"], as_index=False)["volume"].sum()
                 .pivot(index="Família Comercial", columns="mes", values="volume")
                 .reindex(columns=months, fill_value=0.0))
    piv_sales.columns = [MONTHS_PT[m] for m in piv_sales.columns]
    piv_sales = piv_sales.reset_index()

    # TEC MAP
    tmap = _tech_map()
    if tmap.empty:
        piv_sales["Tecnologia"] = "N/D"
    else:
        piv_sales = piv_sales.merge(tmap, on="Família Comercial", how="left")
        piv_sales["Tecnologia"] = piv_sales["Tecnologia"].fillna("N/D")

    # CAPACIDADE
    cap = _capacidade_por_tec(year, months)  # ['Tecnologia', meses_pt]
    if cap.empty:
        cap = pd.DataFrame({"Tecnologia": ["N/D"], **{mc: 0.0 for mc in months_pt}})

    # ESTOQUE INICIAL (primeiro mês YTG)
    ei_raw = _estoque_inicial_por_familia(year)
    ei_raw = ei_raw[ei_raw.get("Ano", year) == year] if "Ano" in ei_raw.columns else ei_raw
    first_m = months[0]
    fam_list = piv_sales["Família Comercial"].tolist()
    ei_map = _apply_estoque_inicial(first_m, fam_list, ei_raw)

    # --------- por tecnologia ----------
    kpis: Dict[str, pd.DataFrame] = {}
    details: Dict[str, pd.DataFrame] = {}

    tecs = sorted(piv_sales["Tecnologia"].dropna().unique().tolist())
    resumo_rows = []

    for tec in tecs:
        sub = piv_sales[piv_sales["Tecnologia"] == tec].reset_index(drop=True)
        fams = sub["Família Comercial"].tolist()

        # Vendas por mês (somatório)
        vend_row = {mc: float(sub[mc].sum()) for mc in months_pt}

        # Capacidade por mês (somada para a tecnologia)
        cap_sub = cap[cap["Tecnologia"] == tec]
        cap_row = {mc: float(pd.to_numeric(cap_sub[mc], errors="coerce").fillna(0).sum()) for mc in months_pt} if not cap_sub.empty else {mc: 0.0 for mc in months_pt}

        # Detalhamento por família (MultiIndex: mês -> indicador)
        cols = pd.MultiIndex.from_product([months_pt, ["Inicial","Produção","Vendas","Estoque"]])
        det = pd.DataFrame(index=fams, columns=cols, data=0.0)

        # preencher vendas por família
        for mc in months_pt:
            det[(mc, "Vendas")] = pd.to_numeric(sub[mc], errors="coerce").fillna(0.0).values

        # dicionários para KPIs (somatórios por mês)
        ei_tec: Dict[str, float] = {}
        need_tec: Dict[str, float] = {}
        cap_tec: Dict[str, float] = {}
        gap_tec: Dict[str, float] = {}

        # mês a mês
        for i, m in enumerate(months):
            mc = MONTHS_PT[m]

            # Estoque inicial do mês por família
            if i == 0:
                ini_fam = np.array([ei_map.get(f, 0.0) for f in fams], dtype=float)
            else:
                prevc = MONTHS_PT[months[i-1]]
                ini_fam = det[(prevc, "Estoque")].to_numpy(dtype=float)

            det[(mc, "Inicial")] = ini_fam

            # Necessidade por família
            vend_fam = det[(mc, "Vendas")].to_numpy(dtype=float)
            need_fam = np.maximum(vend_fam - ini_fam, 0.0)
            need_total = float(need_fam.sum())
            need_tec[mc] = need_total

            # Produção limitada pela capacidade do mês (rateio proporcional à necessidade)
            cap_m = float(cap_row.get(mc, 0.0))
            prod_total = min(need_total, cap_m)
            if need_total > 0:
                prod_fam = need_fam * (prod_total / need_total)
            else:
                prod_fam = np.zeros_like(need_fam)

            det[(mc, "Produção")] = prod_fam

            # Estoque final = Inicial + Produção - Vendas  (pode ser negativo)
            det[(mc, "Estoque")] = ini_fam + prod_fam - vend_fam

            # KPIs tec
            ei_tec[mc] = float(ini_fam.sum())
            cap_tec[mc] = cap_m
            gap_tec[mc] = cap_m - need_total  # negativo = ruptura

        # TOTAL linha
        total = pd.DataFrame(det.sum(axis=0)).T
        total.index = ["TOTAL"]
        det_full = pd.concat([total, det], axis=0)
        det_full.index.name = "Indicador"

        # Oculta linhas totalmente zeradas (exceto TOTAL)
        def _row_nonzero(s: pd.Series) -> bool:
            if s.name == "TOTAL":
                return True
            return (pd.to_numeric(s, errors="coerce").fillna(0) != 0).any()

        det_filtered = det_full[det_full.apply(_row_nonzero, axis=1)]

        # ------- KPIs TEC -------
        kpi_df = pd.DataFrame(
            {
                "Métrica": ["Vendas", "Estoque Inicial", "Necessidade Produção", "Capacidade", "Gap Capacidade"],
                **{
                    mc: [
                        vend_row.get(mc, 0.0),
                        ei_tec.get(mc, 0.0),
                        need_tec.get(mc, 0.0),
                        cap_tec.get(mc, 0.0),
                        gap_tec.get(mc, 0.0),
                    ]
                    for mc in months_pt
                },
            }
        )

        # Guardar p/ resumo consolidado
        resumo_rows.append({
            "Tecnologia": tec,
            "Necessidade (YTG)": sum(need_tec.values()),
            "Capacidade (YTG)": sum(cap_tec.values()),
            "Gap (YTG)": sum(gap_tec.values()),
        })

        kpis[tec] = kpi_df
        details[tec] = det_filtered

    resumo_df = pd.DataFrame(resumo_rows) if resumo_rows else pd.DataFrame(columns=["Tecnologia","Necessidade (YTG)","Capacidade (YTG)","Gap (YTG)"])
    return kpis, details, resumo_df

# ======================================================================================
# INTERFACE
# ======================================================================================

def main():
    st.title("Validação — Tecnologia × Família × Volume")

    with st.sidebar:
        st.subheader("Controles")
        try:
            cur = load_current_long()
            ylist = sorted(pd.to_numeric(cur["ano"], errors="coerce").dropna().astype(int).unique().tolist())
            default_year = ylist[-1] if ylist else pd.Timestamp.today().year
        except Exception:
            default_year = pd.Timestamp.today().year

        year = st.number_input("Ano", value=int(default_year), step=1)
        use_ui = st.toggle("Usar Volumes da UI (YTG)", value=True, help="ON: volumes da UI; OFF: RES_WORKING")
        escala_label = st.selectbox("Escala", options=["1x", "1.000x", "1.000.000x"], index=0)
        scale = {"1x": 1, "1.000x": 1_000, "1.000.000x": 1_000_000}[escala_label]

    months = _ytg_months(year)
    if not months:
        st.info("Não há meses YTG para o ano selecionado (cutoff = Dezembro).")
        st.stop()
    months_pt = [MONTHS_PT[m] for m in months]

    kpis, details, resumo = compute_by_technology(year, use_ui)

    if not kpis:
        st.warning("Sem dados para exibir.")
        show_logs()
        st.stop()

    tabs = st.tabs([*kpis.keys(), "Consolidado"])

    # ----- Render de cada tecnologia -----
    for tname, tab in zip(kpis.keys(), tabs[:-1]):
        with tab:
            st.markdown(f"### 🔧 Tecnologia: **{tname}**")

            # KPIs
            k = kpis[tname].copy()

            def _kpi_style(df):
                styles = pd.DataFrame("", index=df.index, columns=df.columns)
                for c in df.columns:
                    if c == "Métrica":
                        continue
                    styles.loc[df["Métrica"] == "Gap Capacidade", c] = df.loc[df["Métrica"] == "Gap Capacidade", c].map(
                        lambda v: _style_gap(float(v))
                    )
                return styles

            k_fmt = {c: (lambda x, sc=scale: _fmt_int(x, sc)) for c in k.columns if c != "Métrica"}
            st.dataframe(
                k.style.apply(_kpi_style, axis=None).format(k_fmt),
                use_container_width=True,
                hide_index=True,
            )

            # Detalhamento
            st.markdown("#### 🔎 Detalhamento operacional (por família)")
            d = details[tname].copy()

            # remover meses totalmente zerados
            to_drop = []
            for mc in months_pt:
                block = d.loc[:, pd.IndexSlice[mc, :]]
                vals = pd.to_numeric(block.values.reshape(-1), errors="coerce")
                if np.all(np.nan_to_num(vals) == 0):
                    to_drop.append(mc)
            keep_mc = [mc for mc in months_pt if mc not in to_drop]
            if keep_mc:
                cols_keep = []
                for mc in keep_mc:
                    cols_keep.extend([(mc, x) for x in ["Inicial","Produção","Vendas","Estoque"]])
                d = d.loc[:, cols_keep]
            else:
                st.info("Todos os meses do YTG ficaram zerados para esta tecnologia.")
                continue

            # estilos: TOTAL em amarelo; Inicial/Estoque com verde/vermelho condicional
            def _det_style(df):
                styles = pd.DataFrame("", index=df.index, columns=df.columns)
                styles.loc["TOTAL", :] = "background-color:#fff3bf; font-weight:700"
                for col in df.columns:
                    mes, metric = col
                    if metric in ("Inicial", "Estoque"):
                        styles[col] = df[col].map(
                            lambda v: "background-color:#fde2e2; color:#991b1b" if pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0] < 0
                            else ("background-color:#ecfdf5; color:#065f46" if pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0] > 0 else "")
                        )
                return styles

            det_fmt = {c: (lambda x, sc=scale: _fmt_int(x, sc)) for c in d.columns}
            st.dataframe(
                d.style.apply(_det_style, axis=None).format(det_fmt),
                use_container_width=True,
            )

    # ----- Consolidado -----
    with tabs[-1]:
        st.markdown("### 🧩 Consolidado (todas as tecnologias) — YTG")
        if resumo is None or resumo.empty:
            st.info("Sem dados consolidados.")
        else:
            for col in ["Necessidade (YTG)", "Capacidade (YTG)", "Gap (YTG)"]:
                resumo[col] = pd.to_numeric(resumo[col], errors="coerce").fillna(0.0)

            st.dataframe(
                resumo.assign(
                    **{
                        "Necessidade (YTG)": resumo["Necessidade (YTG)"].map(lambda v: _fmt_int(v, scale)),
                        "Capacidade (YTG)": resumo["Capacidade (YTG)"].map(lambda v: _fmt_int(v, scale)),
                        "Gap (YTG)": resumo["Gap (YTG)"].map(lambda v: _fmt_signed(v, scale)),
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            try:
                import altair as alt
                melt = resumo.melt(id_vars="Tecnologia", value_vars=["Necessidade (YTG)","Capacidade (YTG)"])
                chart = (
                    alt.Chart(melt)
                    .mark_bar()
                    .encode(
                        x=alt.X("value:Q", title=f"Unidades ({escala_label})"),
                        y=alt.Y("Tecnologia:N", sort="-x"),
                        color=alt.Color("variable:N", title=""),
                        tooltip=["Tecnologia","variable","value"]
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart, use_container_width=True)
            except Exception as e:
                log(f"ℹ️ Altair indisponível ({e}); gráfico pulado.")

            # Heatmap Gap por mês (com total no topo)
            try:
                heat_rows = []
                for tec, k in kpis.items():
                    for mc in months_pt:
                        gap = float(pd.to_numeric(k.loc[k["Métrica"] == "Gap Capacidade", mc], errors="coerce").fillna(0).values[0])
                        heat_rows.append({"Tecnologia": tec, "Mês": mc, "Gap": gap})
                heat = pd.DataFrame(heat_rows)
                heat_piv = heat.pivot(index="Tecnologia", columns="Mês", values="Gap")
                total_row = pd.DataFrame(heat_piv.sum(axis=0)).T
                total_row.index = ["TOTAL"]
                heat_with_total = pd.concat([total_row, heat_piv], axis=0)

                st.markdown("#### Heatmap — Gap por Mês × Tecnologia")
                st.dataframe(
                    heat_with_total.style.applymap(_style_gap)
                    .format(lambda v: _fmt_signed(v, scale))
                    .apply(lambda s: ["background-color:#fff3bf; font-weight:700" if s.name == "TOTAL" else "" for _ in s], axis=1),
                    use_container_width=True,
                )
            except Exception as e:
                log(f"ℹ️ Heatmap indisponível ({e}).")

    st.divider()
    show_logs()


def show_logs():
    with st.expander("🧾 Diagnóstico & Logs (expandir apenas se necessário)"):
        st.code(json.dumps({"logs": LOGS}, indent=2, ensure_ascii=False))

# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
