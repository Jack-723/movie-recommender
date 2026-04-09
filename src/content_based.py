import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from utils import load_movies, load_all 


for _res in ['punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']: 
    nltk.download(_res, quiet = True)

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words('english'))


def preprocess_text(title, genres) : 

    title = re.sub(r'\(\d{4}\)', '', title)
    genres = genres.replace('|', ' ')

    text = (title + ' ' + genres).lower()
    tokens = word_tokenize(text)

    tokens = [_lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in _stop_words]
    return ' '.join(tokens)

def build_corpus(movies_df):
    # one preprocessed document per movie, preserving movie_id index
    corpus = movies_df.apply(lambda row: preprocess_text(row['title'], row['genres']), axis=1)
    corpus.index = movies_df['movie_id']
    return corpus


def build_bow_matrix(corpus):
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(corpus)
    return bow_matrix, vectorizer

def compute_bow_similarity(bow_matrix, movies_df):
    sim = cosine_similarity(bow_matrix)
    sim_df = pd.DataFrame(sim, index=movies_df['movie_id'], columns=movies_df['movie_id'])
    return sim_df


def recommend_bow(movie_id, movies_df, sim_df, n=10):
    if movie_id not in sim_df.index:
        return pd.DataFrame()

    scores = sim_df[movie_id].drop(movie_id).sort_values(ascending=False).head(n)
    recs = scores.reset_index()
    recs.columns = ['movie_id', 'similarity']
    recs = recs.merge(movies_df[['movie_id', 'title', 'genres']], on='movie_id', how='left')
    return recs[['movie_id', 'title', 'genres', 'similarity']]




def recommend_bow_for_user(user_id, train, movies, sim_df, n=10):
    # get movies this user liked in training data
    liked = train[(train['user_id'] == user_id) & (train['rating'] >= 4)]['movie_id']
    liked = [m for m in liked if m in sim_df.index]

    if not liked:
        return []

    seen = set(train[train['user_id'] == user_id]['movie_id'])

    # aggregate similarity scores across all liked movies
    scores = sim_df.loc[liked].mean(axis=0)
    scores = scores.drop(index=[m for m in liked if m in scores.index])
    scores = scores[~scores.index.isin(seen)]

    return list(scores.sort_values(ascending=False).head(n).index)






def save_outputs(recs_df, out_dir='outputs'):
    os.makedirs(out_dir, exist_ok=True)
    recs_df.to_csv(os.path.join(out_dir, 'bow_recommendations_sample.csv'), index=False)


if __name__ == "__main__":
    from nonpersonalized import time_based_split, evaluate_recommender

    ratings, movies, users = load_all()

    print("Preprocessing movie text...")
    corpus = build_corpus(movies)

    print("Building BoW matrix...")
    bow_matrix, vectorizer = build_bow_matrix(corpus)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    print("Computing cosine similarity...")
    sim_df = compute_bow_similarity(bow_matrix, movies)

    # sample recommendations for Toy Story (movie_id=1)
    sample_id = 1
    sample_title = movies[movies['movie_id'] == sample_id]['title'].values[0]
    print(f"\nTop 10 recommendations for '{sample_title}':")
    recs = recommend_bow(sample_id, movies, sim_df, n=10)
    print(recs.to_string(index=False))

    # evaluation
    print("\nSplitting data...")
    train, test = time_based_split(ratings, test_size=0.2)

    def bow_rec_fn(user_id, train, movies, n=10):
        return recommend_bow_for_user(user_id, train, movies, sim_df, n=n)

    print("Evaluating BoW content-based recommender...")
    np.random.seed(42)
    results = evaluate_recommender(bow_rec_fn, train, test, movies, n=10, sample_size=500)
    print(f"BoW content-based: {results}")

    print("\nSaving outputs...")
    save_outputs(recs)
    print("saved bow_recommendations_sample.csv")