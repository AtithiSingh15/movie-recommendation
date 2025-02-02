import streamlit as st
import pandas as pd
import requests
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the movie dataset
movies_data = pd.read_csv('movies.csv')

# Select relevant features
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combine features
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

# Convert text data into numerical vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Compute similarity scores
similarity = cosine_similarity(feature_vectors)

# Function to fetch movie poster from TMDb API
def get_movie_poster(movie_title):
    api_key = "3bdaa7cb3b280be2be098de84e1527c5"  # Replace with your actual API key
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_title}"

    response = requests.get(search_url).json()
    if response.get("results"):
        poster_path = response["results"][0].get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return None

# Streamlit UI
st.title("🎬 Movie Recommendation System")

# User input
movie_name = st.text_input("Enter your favorite movie:")

if st.button("Get Recommendations"):
    if movie_name:
        list_of_all_titles = movies_data['title'].tolist()
        find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

        if find_close_match:
            close_match = find_close_match[0]
            index_of_the_movie = movies_data[movies_data.title == close_match].index[0]

            similarity_score = list(enumerate(similarity[index_of_the_movie]))
            sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

            st.subheader("Recommended Movies:")
            for i, movie in enumerate(sorted_similar_movies[1:11]):  # Top 10 recommendations
                index = movie[0]
                title_from_index = movies_data.iloc[index]['title']
                poster_url = get_movie_poster(title_from_index)

                # Display movie title and poster
                st.write(f"**{i+1}. {title_from_index}**")
                if poster_url:
                    st.image(poster_url, width=150)
        else:
            st.error("Movie not found. Try another title.")
