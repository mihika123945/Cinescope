import os
import pandas as pd
import numpy as np
import torch
import h5py
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
from sklearn.preprocessing import MultiLabelBinarizer
from datetime import datetime

# File paths
preprocessed_hdf5_file = "preprocessed_data.h5"
movies_file = "movies_data.h5"
features_file = "movie_features.h5"

# --- STEP 4: Prepare Features for Similarity ---
def prepare_features(movie_features):
    # Normalize the features for cosine similarity
    movie_features_tensor = movie_features / movie_features.norm(dim=1, keepdim=True)
    return movie_features_tensor

def get_recommendations_from_precomputed(selected_movies, movies, movie_features_tensor, num_recommendations=5, years_past=None, min_rating=6.0):
    selected_indices = []
    for title in selected_movies:
        if title in movies['primaryTitle'].values:
            selected_indices.append(movies[movies['primaryTitle'] == title].index[0])
        else:
            print(f"'{title}' not found in the dataset. Skipping...")

    if not selected_indices:
        return "No valid movies found in the dataset for the selection."

    # Convert indices to PyTorch tensor
    selected_indices_tensor = torch.tensor(selected_indices, dtype=torch.long)

    # Compute mean vector for selected movies
    mean_vector = movie_features_tensor[selected_indices_tensor].mean(dim=0, keepdim=True)

    # Compute cosine similarity
    similarities = torch.matmul(movie_features_tensor, mean_vector.T).squeeze()
    top_indices = torch.topk(similarities, num_recommendations * 10).indices

    # Filter and process recommendations
    recommendations = movies.iloc[top_indices.tolist()]  # Convert tensor indices to list for Pandas
    if years_past is not None:
        current_year = datetime.now().year
        min_year = current_year - years_past
        recommendations = recommendations[pd.to_numeric(recommendations['startYear'], errors='coerce') >= min_year]

    if min_rating is not None:
        recommendations = recommendations[recommendations['averageRating'] >= min_rating]

    if recommendations.empty:
        print("No recommendations found. Returning fallback recommendations...")
        recommendations = movies.iloc[top_indices.tolist()].sort_values(by='averageRating', ascending=False).head(num_recommendations)
    else:
        recommendations = recommendations.head(num_recommendations)

    return recommendations['primaryTitle'].tolist()

# --- Dash App Setup ---
app = dash.Dash(__name__)
app.title = "Movie Dashboard"

app.layout = html.Div(
    [
        # Header
        html.Div(
            [
                html.H1("Movie Dashboard", style={'textAlign':'center', 'color': '#2c3e50'}),
                html.P("Explore trends and discover personalized movie recommendations!", style={'textAlign': 'center', 'color': '#34495e'}),
            ],
            style={'padding': '20px', 'backgroundColor': '#ecf0f1'}
        ),

        # Visualizations Section
        html.Div(
            [
                html.H2("Trend of Number of Movies and TV Shows", style={'textAlign': 'center', 'color': '#2c3e50'}),
                dcc.Graph(id='trend-graph', style={'margin': '20px auto', 'width': '90%', 'height': '400px'}),
            ],
            style={'margin': '20px 0', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}
        ),
        
        html.Div(
            [
                html.H2("Ratio of Successful Movies and TV Shows", style={'textAlign': 'center', 'color': '#2c3e50'}),
                dcc.Graph(id='successful-ratio-graph', style={'margin': '20px auto', 'width': '90%', 'height': '500px'}),
            ],
            style={'margin': '20px 0', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}
        ),
        
        html.Div(
            [
                html.H2("Genre Distribution (Top 10 genres)", style={'textAlign': 'center', 'color': '#2c3e50'}),
                dcc.Graph(id='genre-distribution-graph', style={'margin': '20px auto', 'width': '90%', 'height': '500px'}),
            ],
            style={'margin': '20px 0', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}
        ),

        html.Div(
            [
                html.H2("Ratings Distribution for Top 10 Genres", style={'textAlign': 'center', 'color': '#2c3e50'}),
                dcc.Graph(id='ratings-by-genre-graph', style={'margin': '20px auto', 'width': '90%', 'height': '500px'}),
            ],
            style={'margin': '20px 0', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}
        ),


        # Recommender System Section
        html.Div(
            [
                html.H2("Recommender System", style={'textAlign': 'center', 'color': '#2c3e50'}),
                html.Label("Enter a Movie You Like:", style={'color': '#34495e', 'fontWeight': 'bold'}),
                dcc.Input(
                    id='movie-input',
                    type='text',
                    placeholder='Enter a movie name...',
                    style={
                        'width': '90%', 'padding': '10px', 'margin': '10px 0', 
                        'borderRadius': '5px', 'border': '1px solid #bdc3c7'
                    }
                ),
                html.Br(),
                html.Label("Number of Recommendations:", style={'color': '#34495e', 'fontWeight': 'bold'}),
                html.Div(
                    dcc.Slider(
                        id='recommendation-slider',
                        min=1,
                        max=20,
                        step=1,
                        value=5,
                        marks={i: str(i) for i in range(1, 21)},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                    style={'margin': '20px 0'}  # Styling applied to the parent Div
                ),

                html.Label("Minimum Rating:", style={'color': '#34495e', 'fontWeight': 'bold'}),
                html.Div(
                    dcc.Slider(
                        id='rating-slider',
                        min=0.0,
                        max=10.0,
                        step=0.5,
                        value=6.0,
                        marks={i: str(i) for i in range(0, 11)},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                    style={'margin': '20px 0'}  # Styling applied to the parent Div
                ),
                html.Label("Years Past (Optional):", style={'color': '#34495e', 'fontWeight': 'bold'}),
                dcc.Input(
                    id='years-past-input',
                    type='number',
                    placeholder='Years past...',
                    style={
                        'width': '90%', 'padding': '10px', 'margin': '10px 0', 
                        'borderRadius': '5px', 'border': '1px solid #bdc3c7'
                    }
                ),
                html.Button(
                    "Get Recommendations", id='recommend-button', n_clicks=0,
                    style={
                        'backgroundColor': '#3498db', 'color': 'white', 'padding': '10px 20px',
                        'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer', 'fontWeight': 'bold',
                        'margin': '20px 0'
                    }
                ),
                html.H3("Recommended Movies:", style={'textAlign': 'center', 'color': '#2c3e50'}),
                html.Ul(id='recommendations-output', style={'listStyleType': 'none', 'padding': '0', 'textAlign': 'center'}),
            ],
            style={
                'margin': '20px 0', 'padding': '20px', 'backgroundColor': '#f8f9fa', 
                'borderRadius': '10px', 'width': '80%', 'margin': 'auto'
            }
        )
    ],
    style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#ecf0f1', 'padding': '20px'}
)

# --- Callbacks ---
@app.callback(
    Output('trend-graph', 'figure'),
    Input('trend-graph', 'id')  # Use its own ID as a placeholder input
)
def update_trend_graph(_):
    fig = px.line(
        grouped,
        x="startYear",
        y="count",
        color="titleType",
        labels={"startYear": "Year", "count": "Count", "titleType": "Type"},
        title="Trend of Movies and TV Series"
    )
    return fig


@app.callback(
    Output('successful-ratio-graph', 'figure'),
    Input('successful-ratio-graph', 'id')
)

def update_successful_ratio_graph(_):
    # Compute total counts
    total_grouped = grouped.groupby(["startYear", "titleType"]).size().reset_index(name="total_count")

    # Compute successful counts (Avg Rating > 7)
    successful_grouped = grouped[grouped["averageRating"] > 7].groupby(["startYear", "titleType"]).size().reset_index(name="successful_count")

    # Merge total and successful counts
    merged_counts = pd.merge(total_grouped, successful_grouped, on=["startYear", "titleType"], how="left")
    merged_counts["successful_count"] = merged_counts["successful_count"].fillna(0)

    # Calculate the ratio
    merged_counts["ratio"] = merged_counts["successful_count"] / merged_counts["total_count"]

    # Create a line plot of the ratios
    fig = px.line(
        merged_counts,
        x="startYear",
        y="ratio",
        color="titleType",
        labels={
            "startYear": "Year",
            "ratio": "Successful/Total Ratio",
            "titleType": "Type"
        },
        title="Ratio of Successful Movies and TV Shows to Total Movies and TV Shows"
    )
    return fig

@app.callback(
    Output('genre-distribution-graph', 'figure'),
    Input('genre-distribution-graph', 'id')
)
def update_genre_distribution_graph(_):
    # Ensure genres column is consistent
    grouped["genres"] = grouped["genres"].fillna("Unknown").astype(str)

    # Clean and standardize genre data
    grouped["genres"] = grouped["genres"].str.replace(r"[^\w,]", "", regex=True)  # Remove special characters
    grouped["genres"] = grouped["genres"].str.split(",")  # Split genres into lists
    genre_data = grouped.explode("genres")  # Explode lists into rows

    # Standardize genre names (strip whitespace, remove empty strings)
    genre_data["genres"] = genre_data["genres"].str.strip()
    genre_data = genre_data[genre_data["genres"] != ""]  # Remove empty genres

    # Count occurrences of each genre
    genre_counts = genre_data["genres"].value_counts().head(10).reset_index()
    genre_counts.columns = ["genre", "count"]

    # Create a bar plot
    fig = px.bar(
        genre_counts,
        x="genre",
        y="count",
        labels={"genre": "Genre", "count": "Count"},
        title="Top 10 Genres by Count"
    )
    fig.update_layout(xaxis={'categoryorder': 'total descending'})
    return fig

@app.callback(
    Output('ratings-by-genre-graph', 'figure'),
    Input('ratings-by-genre-graph', 'id')
)
def update_ratings_by_genre_graph(_):
    # Ensure genres column is consistent
    grouped["genres"] = grouped["genres"].fillna("Unknown").astype(str)
    grouped["genres"] = grouped["genres"].str.replace(r"[^\w,]", "", regex=True)  # Remove special characters
    grouped["genres"] = grouped["genres"].str.split(",")  # Split genres into lists
    genre_data = grouped.explode("genres")  # Explode lists into rows

    # Standardize genre names (strip whitespace, remove empty strings)
    genre_data["genres"] = genre_data["genres"].str.strip()
    genre_data = genre_data[genre_data["genres"] != ""]  # Remove empty genres

    # Get top 10 genres by count
    top_genres = genre_data["genres"].value_counts().head(10).index

    # Filter data for top 10 genres
    top_genre_data = genre_data[genre_data["genres"].isin(top_genres)]

    # Create a box plot for ratings by genre without data points
    fig = px.box(
        top_genre_data,
        x="genres",
        y="averageRating",
        points=None,  # Remove the individual data points
        title="Distribution of Ratings for Top 10 Genres",
        labels={"genres": "Genre", "averageRating": "Average Rating"},
        template="plotly_white"
    )

    # Additional layout adjustments
    fig.update_layout(
        xaxis={'categoryorder': 'total descending'},  # Order genres by frequency
        yaxis_title="Average Rating",
        xaxis_title="Genre"
    )

    return fig


@app.callback(
    Output('recommendations-output', 'children'),
    [Input('recommend-button', 'n_clicks')],
    [
        State('movie-input', 'value'),
        State('recommendation-slider', 'value'),
        State('rating-slider', 'value'),
        State('years-past-input', 'value')
    ]
)
def update_recommendations(n_clicks, movie_input, num_recommendations, min_rating, years_past):
    if n_clicks > 0 and movie_input:
        selected_movies = [movie_input]
        recommendations = get_recommendations_from_precomputed(
            selected_movies, movies, movie_features_tensor, num_recommendations, years_past, min_rating
        )
        if isinstance(recommendations, list):
            return [html.Li(rec) for rec in recommendations]
        return [html.Li(recommendations)]
    return ["Enter a movie and click 'Get Recommendations'!"]

# --- Main Execution ---
if __name__ == '__main__':
    try:
        # Load or preprocess data
        grouped = pd.read_hdf(preprocessed_hdf5_file, key="grouped") if os.path.exists(preprocessed_hdf5_file) else preprocess_and_save()
        movies = pd.read_hdf(movies_file, key="movies") if os.path.exists(movies_file) else None
        movie_features = torch.tensor(h5py.File(features_file, "r")["movie_features"][:], dtype=torch.float32) if os.path.exists(features_file) else None
        
        # Prepare features and tensor
        movie_features_tensor = prepare_features(movie_features)
        
        app.run_server(debug=True)
    except Exception as e:
        print(f"Error: {e}")
