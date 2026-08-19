import altair as alt
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np

# Show the page title and description.
st.set_page_config(page_title="Movies and Animal Data Visualization", page_icon="🎬")
st.title("🎬 Movies and Animal Data Visualization")
st.write(
    """
    This app visualizes data from [The Movie Databases Just click on the widgets below to explore!
    """
)

# Load the movie data from a CSV. We're caching this so it doesn't reload every time the app
# reruns (e.g., if the user interacts with the widgets).
#@st.cache
def load_movie_data():
    try:
        df = pd.read_csv("data/data/movies_genres_summary4.csv")
        return df
    except Exception as e:
        st.error(f"Error loading movie data: {e}")
        return pd.DataFrame()

# Load the species data from a CSV.
#@st.cache
def load_species_data():
    try:
        df = pd.read_csv("data/data/species_strategies2.csv")
        return df
    except Exception as e:
        st.error(f"Error loading species data: {e}")
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

# Ensure the movie DataFrame has the necessary columns
movie_required_columns = ["year", "ActorId", "Name", "MovieId", "Title", "genre", "Country", "gross"]
if validate_columns(movie_df, movie_required_columns):
    # Define all possible genres
    all_genres = ["Romance", "Film-Noir", "Music", "Comedy", "Biography", "Sport", "Drama", "Animation", "Sci-Fi", 
                  "Western", "War", "Adventure", "Musical", "Action", "Horror", "Thriller", "Fantasy", "Mystery", 
                  "Crime", "Family", "History"]

    # Show a multiselect widget with the genres using `st.multiselect`.
    genres = st.multiselect(
        "Genres",
        all_genres,
        ["Action", "Adventure", "Biography", "Comedy", "Drama", "Horror"]
    )

    # Show a slider widget with the years using `st.slider`.
    years = st.slider("Years", 1986, 2016, (2000, 2024))

    # Filter the movie DataFrame based on the widget input and reshape it.
    df_filtered = movie_df[(movie_df["genre"].isin(genres)) & (movie_df["year"].between(years[0], years[1]))]
    
    # Display the data as a table using `st.dataframe`.
    st.dataframe(
        df_filtered[["year", "Title", "genre", "gross"]],
        use_container_width=True,
    )

    # Aggregate the data for the Altair chart.
    df_reshaped = df_filtered.pivot_table(
        index="year", columns="genre", values="gross", aggfunc="sum", fill_value=0
    )
    df_reshaped = df_reshaped.sort_values(by="year", ascending=False)

    # Display the data as an Altair chart using `st.altair_chart`.
    df_chart = pd.melt(
        df_reshaped.reset_index(), id_vars="year", var_name="genre", value_name="gross"
    )
    chart = (
        alt.Chart(df_chart)
        .mark_line()
        .encode(
            x=alt.X("year:N", title="Year"),
            y=alt.Y("gross:Q", title="Gross earnings ($)"),
            color="genre:N",
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.error("Movie data not loaded correctly or missing necessary columns.")

# Ensure the species DataFrame has the necessary columns
species_required_columns = ["species", "protection", "defense", "attack", "feeding", "satisfaction", "sexual_reproduction"]
if validate_columns(species_df, species_required_columns):
    # Show a multiselect widget with the species using `st.multiselect`.
    species = st.multiselect(
        "Species",
        species_df['species'].unique(),
        species_df['species'].unique()[:5]  # Selecting the first 5 species by default
    )

    # Filter the species DataFrame based on the widget input.
    species_filtered = species_df[species_df["species"].isin(species)]
    
    # Display the data as a table using `st.dataframe`.
    st.dataframe(
        species_filtered,
        use_container_width=True,
    )

    # Show a multiselect widget with the animal variables using `st.multiselect`.
    #variables = st.multiselect(
        #"Variables",
        #["protection", "defense", "attack", "feeding", "satisfaction", "sexual_reproduction"],
        #["protection", "defense"]
    #)

    # Show a bar chart for each selected species.
    #if variables:
    #for sp in species:
        #sp_df = species_filtered[species_filtered['species'] == sp]
        #bar_chart = alt.Chart(sp_df).transform_fold(
            #variables,
            #as_=['Variable', 'Value']
        #).mark_bar().encode(
            #x='Variable:N',
            #y='Value:Q',
            #color='Variable:N',
            #tooltip=['species', 'Variable', 'Value']
        #).properties(
            #title=f'Variables for {sp}',
            #height=320,
            #width=640
        #)
    #st.altair_chart(species_filtered, use_container_width=True)
    chart_data = pd.DataFrame(species_filtered, columns=["feeding", "protection", "satisfaction", "species_filtered"])
    st.bar_chart(chart_data)

    # Prepare data for linear regression plot.
    x_var = st.selectbox("Choose X variable for regression", ["feeding", "protection", "defense", "attack", "satisfaction"])
    y_var = st.selectbox("Choose Y variable for regression", ["satisfaction", "feeding", "protection", "defense", "attack"])

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
            'predicted_' + y_var: y_pred.flatten()
        })

        # Display the linear regression chart.
        regression_chart = alt.Chart(regression_df).mark_point().encode(
            x=f'{x_var}:Q',
            y=f'{y_var}:Q'
        ) + alt.Chart(regression_df).mark_line(color='red').encode(
            x=f'{x_var}:Q',
            y=f'predicted_{y_var}:Q'
        )
        st.altair_chart(regression_chart, use_container_width=True)
    else:
        st.error("Selected variables must be numeric for regression analysis.")
else:
    st.error("Species data not loaded correctly or missing necessary columns.")
