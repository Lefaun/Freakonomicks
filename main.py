import altair as alt
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np
import unicodedata
import io

# Show the page title and description.
st.set_page_config(page_title="Wage and Expenses Data Visualization", page_icon="💰")
st.title("💰 Financial Data Visualization - Frakonomics")
st.write(
    """
    Explora o teu extrato bancário: filtra por categoria e ano, vê a evolução
    dos valores ao longo do tempo, e explora a relação entre variáveis numéricas.
    """
)

# ---------------------------------------------------------------------------
# Categorização automática (equivalente ao "genre" do teu ficheiro de filmes).
# Ajusta as palavras-chave conforme os teus próprios movimentos.
# ---------------------------------------------------------------------------
CATEGORY_KEYWORDS = {
    "Incomes": ["TRF VENCIMENTOS", "VENCIMENTOS DOCEN", "SALARIO", "ORDENADO", "REEMBOLSOS IRS", "DEPOSITO"],
    "Outcomes": ["COMPRA", "COMPRAS", "PAGAMENTO"],
    "Taxes": ["IMPOSTO SELO", "IRS", "IVA", "FINANCAS"],
    "Fees": ["COMISSAO", "MANUT CONTA", "MANUTENCAO"],
    "Wages": ["TRF VENCIMENTOS", "VENCIMENTOS DOCEN"],
    "Health": ["FARMACIA", "SOLINCA", "HOSPITAL", "CLINICA", "SEGURANCA MAXI"],
    "Internet": ["MEO", "NOS ", "VODAFONE", "TELE / TV", "ALTICE"],
    "Energy": ["EDP", "GALP", "ENERGIA"],
    "Water": ["SIMAS", "AGUAS", "EPAL"],
    "Insurances": ["SEGURO", "FIDELIDADE", "ALLIANZ", "TRANQUILIDADE", "DOMESTIC AND GENERAL"],
    "Rent": ["RENDA", "ARRENDAMENTO"],
    "Traveling": ["UBER", "RYANAIR", "TRAINLINE", "EXPEDIA", "TML TRANSP", " CP ", "METRO", "BRISA", "LEVANTAMENTO", "ATM "],
    "Food": ["CONTINENTE", "PINGO DOCE", "AUCHAN", "MINI SUPER", "MINIMERCADO", "SUPERMERCADO", "RESTAUR",
             "CANTINHO", "PIZZA", "MCDONALDS", "PADARIA", "PASTELARIA", "CAFE", "SUMINHO"],
    "Shopping": ["AMAZON", "TEMU", "KLARNA", "WORTEN", "IKEA", "ZARA", "EUROSHOP", "SHIEK", "PAPELARIA"],
    "Subscriptions": ["GOOGLE", "MICROSOFT", "SPOTIFY", "NETFLIX", "COURSERA"],
    "Transfers": ["TRANSFERENCIA", "TRF CAIXADIRECTA", "TRF MBWAY", "TRF CXDAPP", "MBWAY"],
}


def strip_accents(text):
    if not isinstance(text, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).upper()


def categorize(descricao):
    desc_norm = strip_accents(descricao)
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(strip_accents(kw) in desc_norm for kw in keywords):
            return category
    return "Other"


def to_number(series):
    return (
        series.astype(str).str.strip()
        .replace({"": "0", "nan": "0", "None": "0"})
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )


def read_bank_csv(path):
    """Lê o CSV do banco, ignorando linhas de metadados antes da tabela real
    (nome da conta, período, etc.) que os exports da Caixa costumam ter."""
    encodings_to_try = ["cp1252", "latin1", "utf-8-sig"]
    raw_text = None
    used_encoding = None
    for enc in encodings_to_try:
        try:
            with open(path, "r", encoding=enc) as f:
                raw_text = f.read()
            used_encoding = enc
            break
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    if raw_text is None:
        raise FileNotFoundError(f"Não consegui abrir {path}")

    lines = raw_text.splitlines()

    # Encontra a linha do cabeçalho real da tabela (contém "Data mov")
    header_idx = None
    for i, line in enumerate(lines):
        if "DATA MOV" in strip_accents(line):
            header_idx = i
            break
    if header_idx is None:
        header_idx = 0  # fallback: assume que não há metadados

    header_line = lines[header_idx]
    # Detecta o separador mais provável a partir da linha de cabeçalho
    sep = ";" if header_line.count(";") >= header_line.count(",") else ","

    df = pd.read_csv(
        io.StringIO("\n".join(lines[header_idx:])),
        sep=sep,
        engine="python",
        on_bad_lines="skip",
    )
    return df


def prepare_df(df):
    """Limpa e enriquece o dataframe cru do banco."""
    df.columns = [c.strip() for c in df.columns]
    rename_map = {}
    for col in df.columns:
        key = strip_accents(col)
        if "DATA MOV" in key:
            rename_map[col] = "Data mov."
        elif "DESCRI" in key:
            rename_map[col] = "Descrição"
        elif key.startswith("DEBITO"):
            rename_map[col] = "Débito"
        elif key.startswith("CREDITO"):
            rename_map[col] = "Crédito"
        elif "SALDO CONTAB" in key:
            rename_map[col] = "Saldo contabilístico"
        elif "SALDO DISPON" in key:
            rename_map[col] = "Saldo disponível"
    df = df.rename(columns=rename_map)

    df["Data mov."] = pd.to_datetime(df["Data mov."], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Data mov."])

    for col in ["Débito", "Crédito", "Saldo contabilístico", "Saldo disponível"]:
        if col in df.columns:
            df[col] = to_number(df[col].fillna(0))
        else:
            df[col] = 0.0

    df["Valor"] = df["Crédito"] - df["Débito"]
    df["year"] = df["Data mov."].dt.year
    df["genre"] = df["Descrição"].apply(categorize)  # equivalente ao "genre" dos filmes
    return df


# Load the bank data from a CSV. We're caching this so it doesn't reload every time the app
# reruns (e.g., if the user interacts with the widgets).
@st.cache_data
def load_movie_data():
    try:
        df = read_bank_csv("comprovativo_banco.csv")
        return prepare_df(df)
    except Exception as e:
        st.error(f"Error loading transaction data: {e}")
        return pd.DataFrame()


# Load the same data again for the second section (equivalente ao "species_df").
@st.cache_data
def load_species_data():
    try:
        df = read_bank_csv("comprovativo_banco.csv")
        return prepare_df(df)
    except Exception as e:
        st.error(f"Error loading transaction data: {e}")
        return pd.DataFrame()


movie_df = load_movie_data()
species_df = load_species_data()


# Function to check if DataFrame has required columns
def validate_columns(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"DataFrame is missing required columns: {', '.join(missing_columns)}")
        return False
    return True


# ---------------------------------------------------------------------------
# SECÇÃO 1 — equivalente à secção de filmes: categoria x ano
# ---------------------------------------------------------------------------
movie_required_columns = ["year", "Descrição", "genre", "Valor"]
if validate_columns(movie_df, movie_required_columns):
    # Define all possible categories (equivalente aos "genres")
    all_genres = list(CATEGORY_KEYWORDS.keys()) + ["Other"]

    # Show a multiselect widget with the categories using `st.multiselect`.
    genres = st.multiselect(
        "Categorias",
        all_genres,
        ["Incomes", "Outcomes", "Rent", "Food", "Internet", "Energy", "Water"],
    )

    # Show a slider widget with the years using `st.slider`.
    min_year, max_year = int(movie_df["year"].min()), int(movie_df["year"].max())
    years = st.slider("Anos", min_year, max_year, (min_year, max_year))

    # Filter the DataFrame based on the widget input and reshape it.
    df_filtered = movie_df[
        (movie_df["genre"].isin(genres)) & (movie_df["year"].between(years[0], years[1]))
    ]

    # Display the data as a table using `st.dataframe`.
    st.dataframe(
        df_filtered[["year", "Data mov.", "Descrição", "genre", "Valor"]].rename(
            columns={"genre": "Categoria"}
        ),
        use_container_width=True,
    )

    # Aggregate the data for the Altair chart.
    df_reshaped = df_filtered.pivot_table(
        index="year", columns="genre", values="Valor", aggfunc="sum", fill_value=0
    )
    df_reshaped = df_reshaped.sort_values(by="year", ascending=False)

    # Display the data as an Altair chart using `st.altair_chart`.
    df_chart = pd.melt(
        df_reshaped.reset_index(), id_vars="year", var_name="genre", value_name="Valor"
    )
    chart = (
        alt.Chart(df_chart)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:N", title="Ano"),
            y=alt.Y("Valor:Q", title="Valor total (€)"),
            color=alt.Color("genre:N", title="Categoria"),
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.error("Transaction data not loaded correctly or missing necessary columns.")

st.divider()

# ---------------------------------------------------------------------------
# SECÇÃO 2 — equivalente à secção de espécies: categoria x variáveis numéricas
# ---------------------------------------------------------------------------
species_required_columns = ["genre", "Débito", "Crédito", "Saldo contabilístico", "Saldo disponível", "Valor"]
if validate_columns(species_df, species_required_columns):
    # Show a multiselect widget with the categories using `st.multiselect`.
    species = st.multiselect(
        "Categorias para análise",
        species_df["genre"].unique(),
        list(species_df["genre"].unique()[:5]),
    )

    # Filter the DataFrame based on the widget input.
    species_filtered = species_df[species_df["genre"].isin(species)]

    # Display the data as a table using `st.dataframe`.
    st.dataframe(
        species_filtered[["Data mov.", "Descrição", "genre", "Débito", "Crédito", "Valor"]].rename(
            columns={"genre": "Categoria"}
        ),
        use_container_width=True,
    )

    # Show a bar chart with the total value per category.
    chart_data = (
        species_filtered.groupby("genre", as_index=False)[["Débito", "Crédito"]].sum()
        .rename(columns={"genre": "Categoria"})
        .set_index("Categoria")
    )
    st.bar_chart(chart_data)

    # Prepare data for linear regression plot.
    numeric_vars = ["Débito", "Crédito", "Saldo contabilístico", "Saldo disponível"]
    x_var = st.selectbox("Escolhe a variável X para a regressão", numeric_vars, index=0)
    y_var = st.selectbox("Escolhe a variável Y para a regressão", numeric_vars, index=2)

    # Ensure the selected variables are numeric
    if species_filtered[x_var].dtype in [np.float64, np.int64] and species_filtered[y_var].dtype in [np.float64, np.int64]:
        x = np.array(species_filtered[x_var]).reshape(-1, 1)
        y = np.array(species_filtered[y_var]).reshape(-1, 1)

        # Fit the regression model.
        model = LinearRegression()
        model.fit(x, y)
        y_pred = model.predict(x)

        # Create a DataFrame with the regression results.
        regression_df = pd.DataFrame({
            x_var: species_filtered[x_var],
            y_var: species_filtered[y_var],
            "predicted_" + y_var: y_pred.flatten(),
        })

        # Display the linear regression chart.
        regression_chart = alt.Chart(regression_df).mark_point().encode(
            x=f"{x_var}:Q",
            y=f"{y_var}:Q",
        ) + alt.Chart(regression_df).mark_line(color="red").encode(
            x=f"{x_var}:Q",
            y=f"predicted_{y_var}:Q",
        )
        st.altair_chart(regression_chart, use_container_width=True)
    else:
        st.error("Selected variables must be numeric for regression analysis.")
else:
    st.error("Transaction data not loaded correctly or missing necessary columns.")
