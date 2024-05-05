from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import numpy as np

app = Flask(__name__, template_folder='templates')

# Your TMDb API key
API_KEY = "b968b9f7a543c09d79e4d87f7b8effc0"

# Load datasets
movies = pd.read_csv("movies.csv")
# Select relevant columns
movies = movies[['movieId', 'title', 'genres']]

# Split genres using .loc to avoid SettingWithCopyWarning
movies.loc[:, 'genres'] = movies['genres'].str.split('|')

# Convert lists of strings to lowercase and remove whitespaces
movies['genres'] = movies['genres'].apply(lambda x: ' '.join(x).lower().strip())

# Vectorize the text data
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(movies['genres']).toarray()

# Calculate similarity matrix
similarity = cosine_similarity(vector)

# Define the recommendation function
def recommend(movie_title, cosine_sim=similarity):
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:9]
    recommended_movies = [movies.iloc[i[0]]['title'] for i in sim_scores]
    return recommended_movies

def get_movie_poster(movie_title):
    # Find the movieId corresponding to the given movie title
    movie_id_series = movies.loc[movies['title'] == movie_title, 'movieId']
    if len(movie_id_series) == 0:
        return f"No movie found with the title '{movie_title}'"

    movie_id = movie_id_series.iloc[0]

    # Load the links dataset to get the tmdbId for the movie
    links = pd.read_csv("links.csv")
    tmdb_id_series = links.loc[links['movieId'] == movie_id, 'tmdbId']
    if len(tmdb_id_series) == 0:
        return f"No TMDb ID found for the movie '{movie_title}'"

    tmdb_id = tmdb_id_series.iloc[0]

    # Fetch movie details using TMDb API
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"

    # Return None if no poster found
    return None

def popularity_recommendation():
    try:
        # Read datasets
        ratings = pd.read_csv('ratings.csv')

        # Merge ratings with movies to get movie titles
        ratingsAndMovies = ratings.merge(movies, on='movieId')

        # Group by movie title and count the number of ratings
        countRatingsDf = ratingsAndMovies.groupby('title').count()['rating'].reset_index()
        countRatingsDf.rename(columns={'rating': 'numberOfRatings'}, inplace=True)

        # Calculate the mean rating for each movie
        averageRatingsDf = ratingsAndMovies.groupby('title')['rating'].mean().reset_index()
        averageRatingsDf.rename(columns={'rating': 'averageOfRatings'}, inplace=True)

        # Merge count and average rating DataFrames
        popularityDf = countRatingsDf.merge(averageRatingsDf)

        # Filter movies with at least 200 ratings
        popularityDf = popularityDf[popularityDf['numberOfRatings'] >= 200]

        # Sort movies by average rating in descending order
        popularityDf = popularityDf.sort_values('averageOfRatings', ascending=False)

        # Return top 10 popular movies
        popular_movies = popularityDf.head(12)['title'].tolist()
        return popular_movies
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return []

def collaborative_recommendation(movie_title):
    ratings = pd.read_csv('ratings.csv')

    # Merge ratings with movies to get movie titles
    ratingsAndMovies = ratings.merge(movies, on='movieId')

    # Filter users with more than 100 ratings
    x = ratingsAndMovies.groupby('userId').count()['rating'] > 10
    filmGeeks = x[x].index
    userFilteredRatings = ratingsAndMovies[ratingsAndMovies['userId'].isin(filmGeeks)]

    # Filter movies with at least 50 ratings
    y = userFilteredRatings.groupby('title').count()['rating'] >= 5
    highRatedMovies = y[y].index
    finalRatings = userFilteredRatings[userFilteredRatings['title'].isin(highRatedMovies)]

    # Create pivot table
    pt = finalRatings.pivot_table(index='title', columns='userId', values='rating')
    pt.fillna(0, inplace=True)

    if pt.empty:
        return f"No data available for collaborative filtering."

    if movie_title not in pt.index:
        return f"No ratings found for the movie '{movie_title}'."

    index = np.where(pt.index == movie_title)[0][0]
    similarityScores = cosine_similarity(pt)
    similarItems = sorted(list(enumerate(similarityScores[index])), key=lambda x: x[1], reverse=True)[1:9]
    recommended_movies = [pt.index[i[0]] for i in similarItems if i[1] > 0]  # Filter out movies with zero similarity
    if not recommended_movies:
        return "No recommendations available for the provided movie."
    return recommended_movies


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/popularity')
def popularity_recommendation_route():
    popular_movies = popularity_recommendation()
    return render_template('popularity.html', popular_movies=popular_movies, get_movie_poster=get_movie_poster)

@app.route('/collaborative', methods=['GET', 'POST'])
def collaborative_recommendation_route():
    if request.method == 'POST':
        movie_title = request.form['movie_title']
        collaborative_movies = collaborative_recommendation(movie_title)
        return render_template('collaborative.html', collaborative_movies=collaborative_movies, get_movie_poster=get_movie_poster)
    return render_template('collaborative_input.html')

@app.route('/content-based', methods=['GET', 'POST'])
def content_based_recommendation():
    if request.method == 'POST':
        movie_title = request.form['movie_title']
        recommended_movies = recommend(movie_title)
        return render_template('content_based.html', recommended_movies=recommended_movies, get_movie_poster=get_movie_poster)
    return render_template('content_based.html')

if __name__ == '__main__':
    app.run(debug=True)
