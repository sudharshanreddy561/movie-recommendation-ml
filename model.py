import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies = pd.read_csv("movies.csv")

# Convert genres to vectors
vectorizer = TfidfVectorizer()
genre_matrix = vectorizer.fit_transform(movies["genre"])

# Similarity matrix
similarity = cosine_similarity(genre_matrix)

def recommend_movie(movie_title):
    if movie_title not in movies["title"].values:
        print("Movie not found")
        return

    index = movies[movies["title"] == movie_title].index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print(f"\n🎬 Recommendations for '{movie_title}':")
    for i in scores[1:4]:
        print("-", movies.iloc[i[0]]["title"])

# Test
recommend_movie("Inception")
