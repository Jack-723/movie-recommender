import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from utils import load_movies, load_all

# load model once at module level — downloads ~90MB on first run, then cached
model = SentenceTransformer('all-MiniLM-L6-v2')


def build_bert_corpus(movies_df):
    # strip year from title, replace genre separator — no lemmatization needed
    def clean(row):
        title = re.sub(r'\(\d{4}\)', '', row['title']).strip()
        genres = row['genres'].replace('|', ' ')
        return title + ' ' + genres

    corpus = movies_df.apply(clean, axis=1)
    corpus.index = movies_df['movie_id']
    return corpus


def compute_bert_embeddings(corpus):
    # encode all movie texts into dense 384-dim vectors
    print("Encoding movies with BERT (this may take a minute)...")
    embeddings = model.encode(corpus.tolist(), show_progress_bar=True, batch_size=64)
    return embeddings


def compute_bert_similarity(embeddings, movies_df):
    sim = cosine_similarity(embeddings)
    sim_df = pd.DataFrame(sim, index=movies_df['movie_id'], columns=movies_df['movie_id'])
    return sim_df


def recommend_bert(movie_id, movies_df, sim_df, n=10):
    if movie_id not in sim_df.index:
        return pd.DataFrame()

    scores = sim_df[movie_id].drop(movie_id).sort_values(ascending=False).head(n)
    recs = scores.reset_index()
    recs.columns = ['movie_id', 'similarity']
    recs = recs.merge(movies_df[['movie_id', 'title', 'genres']], on='movie_id', how='left')
    return recs[['movie_id', 'title', 'genres', 'similarity']]


def recommend_bert_for_user(user_id, train, movies, sim_df, n=10):
    liked = train[(train['user_id'] == user_id) & (train['rating'] >= 4)]['movie_id']
    liked = [m for m in liked if m in sim_df.index]

    if not liked:
        return []

    seen = set(train[train['user_id'] == user_id]['movie_id'])
    scores = sim_df.loc[liked].mean(axis=0)
    scores = scores.drop(index=[m for m in liked if m in scores.index])
    scores = scores[~scores.index.isin(seen)]

    return list(scores.sort_values(ascending=False).head(n).index)


def save_outputs(recs_df, out_dir='outputs'):
    os.makedirs(out_dir, exist_ok=True)
    recs_df.to_csv(os.path.join(out_dir, 'bert_recommendations_sample.csv'), index=False)


if __name__ == "__main__":
    from nonpersonalized import time_based_split, evaluate_recommender

    ratings, movies, users = load_all()

    corpus = build_bert_corpus(movies)
    embeddings = compute_bert_embeddings(corpus)

    print("Computing cosine similarity...")
    sim_df = compute_bert_similarity(embeddings, movies)

    # sample recommendations for Toy Story (movie_id=1)
    sample_id = 1
    sample_title = movies[movies['movie_id'] == sample_id]['title'].values[0]
    print(f"\nTop 10 recommendations for '{sample_title}':")
    recs = recommend_bert(sample_id, movies, sim_df, n=10)
    print(recs.to_string(index=False))

    # evaluation
    print("\nSplitting data...")
    train, test = time_based_split(ratings, test_size=0.2)

    def bert_rec_fn(user_id, train, movies, n=10):
        return recommend_bert_for_user(user_id, train, movies, sim_df, n=n)

    print("Evaluating BERT content-based recommender...")
    np.random.seed(42)
    results = evaluate_recommender(bert_rec_fn, train, test, movies, n=10, sample_size=500)
    print(f"BERT content-based: {results}")

    print("\nSaving outputs...")
    save_outputs(recs)
    print("saved bert_recommendations_sample.csv")