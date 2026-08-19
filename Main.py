import io
import re
import unicodedata
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Dashboard Financeiro", layout="wide")

DEFAULT_PATH = "comprovativo-banco.csv"

# ---------------------------------------------------------------------------
# 1) DICIONÁRIO DE CATEGORIAS
#    Ajusta / acrescenta palavras-chave conforme os teus movimentos.
#    A comparação é feita em maiúsculas e sem acentos, por isso não precisas
#    de te preocupar com "Á" vs "A".
# ---------------------------------------------------------------------------
CATEGORY_KEYWORDS = {
    "Incomes": [
        "TRF VENCIMENTOS", "VENCIMENTOS DOCEN", "SALARIO", "ORDENADO",
        "REEMBOLSOS IRS", "DEPOSITO", "TRF REEMBOLSOS",
    ],
    "Rent": ["RENDA", "ARRENDAMENTO"],
    "Internet": ["MEO", "NOS ", "VODAFONE", "TELE / TV", "TELECOM", "ALTICE"],
    "Energy": ["EDP", "GALP", "ENERGIA", "ELECTRICIDADE"],
    "Water": ["SIMAS", "AGUAS", "EPAL"],
    "Insurances": [
        "SEGURO", "FIDELIDADE", "ALLIANZ", "TRANQUILIDADE", "DOMESTIC AND GENERAL",
    ],
    "Taxes": ["IMPOSTO SELO", "IRS", "IVA", "FINANCAS", "AT AUTORIDADE"],
    "Fees": [
        "COMISSAO", "MANUT CONTA", "MANUTENCAO", "COM FORA Z EURO",
    ],
    "Traveling": [
        "UBER", "RYANAIR", "TRAINLINE", "EXPEDIA", "TML TRANSP", " CP ",
        "METRO", "BRISA", "TAP ", "ATM ", "LEVANTAMENTO",
    ],
    "Food": [
        "CONTINENTE", "PINGO DOCE", "AUCHAN", "MINI SUPER", "MINIMERCADO",
        "SUPERMERCADO", "RESTAUR", "CANTINHO", "PIZZA", "MCDONALDS",
        "PADARIA", "PASTELARIA", "CAFE", "TASCA", "SUMINHO", "INTERMEMARTINS",
        "FRUTARIAS", "TELEPIZZA", "SAVANA SUSHI", "COPENHAGEN COF",
    ],
    "Health": [
        "FARMACIA", "SOLINCA", "HOSPITAL", "CLINICA", "SEGURANCA MAXI",
    ],
    "Shopping": [
        "AMAZON", "TEMU", "KLARNA", "WORTEN", "IKEA", "ZARA", "EUROSHOP",
        "SHIEK", "PAPELARIA", "DECATHLON", "FNAC",
    ],
    "Subscriptions": [
        "GOOGLE", "MICROSOFT", "SPOTIFY", "NETFLIX", "COURSERA", "DICE.FM",
    ],
    "Transfers": ["TRANSFERENCIA", "TRF CAIXADIRECTA", "TRF MBWAY", "TRF CXDAPP", "MBWAY"],
}

CATEGORY_LIST = list(CATEGORY_KEYWORDS.keys()) + ["Other"]

MESES_PT = {
    1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez",
}


def strip_accents(text: str) -> str:
    if not isinstance(text, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).upper()


def categorize(descricao: str) -> str:
    desc_norm = strip_accents(descricao)
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if strip_accents(kw) in desc_norm:
                return category
    return "Other"


def to_number(series: pd.Series) -> pd.Series:
    """Converte strings tipo '1.234,56' ou '1234,56' para float."""
    return (
        series.astype(str)
        .str.strip()
        .replace({"": "0", "nan": "0", "None": "0"})
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )


@st.cache_data
def load_data(file_bytes: bytes) -> pd.DataFrame:
    # Tenta várias combinações de encoding / separador, comum em exports
    # bancários portugueses (CGD, Millennium, etc.)
    attempts = [
        {"encoding": "utf-8-sig", "sep": ";"},
        {"encoding": "cp1252", "sep": ";"},
        {"encoding": "latin1", "sep": ";"},
        {"encoding": "utf-8-sig", "sep": ","},
        {"encoding": "cp1252", "sep": ","},
    ]
    df = None
    last_err = None
    for opt in attempts:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), **opt)
            if df.shape[1] > 1:
                break
        except Exception as e:  # noqa: BLE001
            last_err = e
            df = None
    if df is None:
        raise ValueError(f"Não consegui ler o CSV automaticamente: {last_err}")

    df.columns = [c.strip() for c in df.columns]

    # Normaliza nomes de colunas esperadas (lida com acentos corrompidos)
    rename_map = {}
    for col in df.columns:
        key = strip_accents(col)
        if "DATA MOV" in key:
            rename_map[col] = "Data mov."
        elif "DATA VALOR" in key:
            rename_map[col] = "Data valor"
        elif "DESCRI" in key:
            rename_map[col] = "Descrição"
        elif key.startswith("DEBITO") or key == "DEBITO":
            rename_map[col] = "Débito"
        elif key.startswith("CREDITO") or key == "CREDITO":
            rename_map[col] = "Crédito"
        elif "SALDO CONTAB" in key:
            rename_map[col] = "Saldo contabilístico"
        elif "SALDO DISPON" in key:
            rename_map[col] = "Saldo disponível"
        elif "CATEGORIA" in key:
            rename_map[col] = "Categoria banco"
    df = df.rename(columns=rename_map)

    required = ["Data mov.", "Descrição"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltam colunas essenciais no CSV: {missing}. "
            f"Colunas encontradas: {list(df.columns)}"
        )

    # Datas (formato português dd-mm-yyyy ou dd/mm/yyyy)
    df["Data mov."] = pd.to_datetime(df["Data mov."], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Data mov."])

    for col in ["Débito", "Crédito"]:
        if col in df.columns:
            df[col] = to_number(df[col].fillna(0))
        else:
            df[col] = 0.0

    df["Valor"] = df["Crédito"] - df["Débito"]
    df["Tipo"] = df["Valor"].apply(lambda v: "Crédito" if v >= 0 else "Débito")

    df["Categoria"] = df["Descrição"].apply(categorize)

    df["Ano"] = df["Data mov."].dt.year
    df["Mes_num"] = df["Data mov."].dt.month
    df["Mes"] = df["Mes_num"].map(MESES_PT)
    df["AnoMes"] = df["Data mov."].dt.to_period("M").astype(str)

    return df.sort_values("Data mov.")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("💰 Dashboard Financeiro")
st.caption("Carregado a partir do extrato bancário — categorização automática por palavra-chave na descrição.")

uploaded = st.sidebar.file_uploader("Carrega o teu comprovativo-banco.csv", type=["csv"])

file_bytes = None
if uploaded is not None:
    file_bytes = uploaded.getvalue()
else:
    try:
        with open(DEFAULT_PATH, "rb") as f:
            file_bytes = f.read()
        st.sidebar.info(f"A usar ficheiro local: {DEFAULT_PATH}")
    except FileNotFoundError:
        pass

if file_bytes is None:
    st.warning("⬅️ Carrega o ficheiro `comprovativo-banco.csv` na barra lateral para começar.")
    st.stop()

try:
    df = load_data(file_bytes)
except Exception as e:  # noqa: BLE001
    st.error(f"Erro ao processar o CSV: {e}")
    st.stop()

# --- Filtros -----------------------------------------------------------
st.sidebar.header("Filtros")

categorias_disponiveis = sorted(df["Categoria"].unique().tolist())
categorias_selecionadas = st.sidebar.multiselect(
    "Categorias",
    categorias_disponiveis,
    default=categorias_disponiveis,
)

anos_disponiveis = sorted(df["Ano"].unique().tolist())
anos_selecionados = st.sidebar.multiselect(
    "Anos",
    anos_disponiveis,
    default=anos_disponiveis,
)

meses_disponiveis = [MESES_PT[m] for m in sorted(df["Mes_num"].unique().tolist())]
meses_selecionados = st.sidebar.multiselect(
    "Meses",
    meses_disponiveis,
    default=meses_disponiveis,
)

tipo_selecionado = st.sidebar.radio("Tipo de movimento", ["Todos", "Crédito", "Débito"], horizontal=False)

min_date, max_date = df["Data mov."].min(), df["Data mov."].max()
date_range = st.sidebar.slider(
    "Intervalo de datas",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
)

# --- Aplica filtros ------------------------------------------------------
df_filtered = df[
    (df["Categoria"].isin(categorias_selecionadas))
    & (df["Ano"].isin(anos_selecionados))
    & (df["Mes"].isin(meses_selecionados))
    & (df["Data mov."] >= pd.Timestamp(date_range[0]))
    & (df["Data mov."] <= pd.Timestamp(date_range[1]))
]

if tipo_selecionado != "Todos":
    df_filtered = df_filtered[df_filtered["Tipo"] == tipo_selecionado]

if df_filtered.empty:
    st.info("Sem movimentos para os filtros selecionados.")
    st.stop()

# --- KPIs ------------------------------------------------------------
total_credito = df_filtered.loc[df_filtered["Valor"] > 0, "Valor"].sum()
total_debito = -df_filtered.loc[df_filtered["Valor"] < 0, "Valor"].sum()
saldo = total_credito - total_debito

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total entradas", f"{total_credito:,.2f} €")
k2.metric("Total saídas", f"{total_debito:,.2f} €")
k3.metric("Saldo (entradas - saídas)", f"{saldo:,.2f} €")
k4.metric("Nº movimentos", f"{len(df_filtered)}")

st.divider()

# --- Gráficos ----------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Gastos por categoria")
    gastos_cat = (
        df_filtered[df_filtered["Valor"] < 0]
        .assign(Valor=lambda d: -d["Valor"])
        .groupby("Categoria", as_index=False)["Valor"]
        .sum()
        .sort_values("Valor", ascending=False)
    )
    fig_bar = px.bar(gastos_cat, x="Categoria", y="Valor", text_auto=".2s")
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.subheader("Distribuição de gastos")
    fig_pie = px.pie(gastos_cat, names="Categoria", values="Valor", hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)

st.subheader("Evolução mensal por categoria")
evolucao = (
    df_filtered[df_filtered["Valor"] < 0]
    .assign(Valor=lambda d: -d["Valor"])
    .groupby(["AnoMes", "Categoria"], as_index=False)["Valor"]
    .sum()
    .sort_values("AnoMes")
)
fig_line = px.bar(evolucao, x="AnoMes", y="Valor", color="Categoria", barmode="stack")
st.plotly_chart(fig_line, use_container_width=True)

st.subheader("Entradas vs Saídas por mês")
fluxo = (
    df_filtered.assign(TipoFluxo=lambda d: d["Valor"].apply(lambda v: "Entradas" if v >= 0 else "Saídas"))
    .assign(ValorAbs=lambda d: d["Valor"].abs())
    .groupby(["AnoMes", "TipoFluxo"], as_index=False)["ValorAbs"]
    .sum()
)
fig_fluxo = px.bar(fluxo, x="AnoMes", y="ValorAbs", color="TipoFluxo", barmode="group")
st.plotly_chart(fig_fluxo, use_container_width=True)

# --- Tabela ------------------------------------------------------------
st.subheader("Movimentos filtrados")
st.dataframe(
    df_filtered[
        ["Data mov.", "Descrição", "Categoria", "Débito", "Crédito", "Valor"]
    ].reset_index(drop=True),
    use_container_width=True,
)

# --- Descrições não categorizadas (ajuda a melhorar o dicionário) -----
outros = df_filtered[df_filtered["Categoria"] == "Other"]
if not outros.empty:
    with st.expander(f"⚠️ {outros['Descrição'].nunique()} descrições caíram em 'Other' — considera adicionar palavras-chave"):
        st.dataframe(outros["Descrição"].value_counts().reset_index().rename(
            columns={"index": "Descrição", "Descrição": "Ocorrências"}
        ))

st.caption(
    "Dica: edita o dicionário CATEGORY_KEYWORDS no topo do ficheiro app.py para "
    "ajustar as categorias às tuas próprias despesas."
)
