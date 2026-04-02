import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils import load_all
from nonpersonalized import time_based_split


def build_user_item_matrix(train_ratings):
    matrix = train_ratings.pivot_table(
        index='user_id',
        columns='movie_id',
        values='rating'
    )
    return matrix


def compute_user_similarity(user_item_matrix):
    filled_matrix = user_item_matrix.fillna(0)
    similarity = cosine_similarity(filled_matrix)
    similarity_df = pd.DataFrame(
        similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )
    return similarity_df


def compute_item_similarity(user_item_matrix):
    filled_matrix = user_item_matrix.fillna(0)
    similarity = cosine_similarity(filled_matrix.T)
    similarity_df = pd.DataFrame(
        similarity,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )
    return similarity_df


def predict_user_user(user_id, movie_id, user_item_matrix, user_similarity, k=10):
    if movie_id not in user_item_matrix.columns:
        return np.nan
    if user_id not in user_item_matrix.index:
        return np.nan

    movie_ratings = user_item_matrix[movie_id]
    similar_users = user_similarity[user_id].drop(user_id).sort_values(ascending=False)

    neighbors = []
    for neighbor_id, sim_score in similar_users.items():
        rating = movie_ratings.get(neighbor_id)
        if not pd.isna(rating):
            neighbors.append((neighbor_id, sim_score, rating))
        if len(neighbors) == k:
            break

    if not neighbors:
        return np.nan

    numerator = sum(sim * rating for _, sim, rating in neighbors)
    denominator = sum(abs(sim) for _, sim, _ in neighbors)

    if denominator == 0:
        return np.nan

    return numerator / denominator


def predict_item_item(user_id, movie_id, user_item_matrix, item_similarity, k=10):
    if movie_id not in user_item_matrix.columns:
        return np.nan
    if user_id not in user_item_matrix.index:
        return np.nan

    user_ratings = user_item_matrix.loc[user_id]
    rated_movies = user_ratings[user_ratings.notna()].index

    similar_items = item_similarity[movie_id].drop(movie_id).sort_values(ascending=False)

    neighbors = []
    for neighbor_movie in similar_items.index:
        if neighbor_movie in rated_movies:
            sim_score = similar_items[neighbor_movie]
            rating = user_ratings[neighbor_movie]
            neighbors.append((neighbor_movie, sim_score, rating))
        if len(neighbors) == k:
            break

    if not neighbors:
        return np.nan

    numerator = sum(sim * rating for _, sim, rating in neighbors)
    denominator = sum(abs(sim) for _, sim, _ in neighbors)

    if denominator == 0:
        return np.nan

    return numerator / denominator


def recommend_user_user(user_id, user_item_matrix, user_similarity, movies_df, top_n=10, k=10):
    if user_id not in user_item_matrix.index:
        return pd.DataFrame()

    user_ratings = user_item_matrix.loc[user_id]
    unseen_movies = user_ratings[user_ratings.isna()].index

    predictions = []
    for movie_id in unseen_movies:
        pred = predict_user_user(user_id, movie_id, user_item_matrix, user_similarity, k=k)
        if not pd.isna(pred):
            predictions.append((movie_id, pred))

    recs = pd.DataFrame(predictions, columns=['movie_id', 'predicted_rating'])
    recs = recs.sort_values(by='predicted_rating', ascending=False).head(top_n)
    recs = recs.merge(movies_df[['movie_id', 'title']], on='movie_id', how='left')

    return recs[['movie_id', 'title', 'predicted_rating']]


def recommend_item_item(user_id, user_item_matrix, item_similarity, movies_df, top_n=10, k=10):
    if user_id not in user_item_matrix.index:
        return pd.DataFrame()

    user_ratings = user_item_matrix.loc[user_id]
    unseen_movies = user_ratings[user_ratings.isna()].index

    predictions = []
    for movie_id in unseen_movies:
        pred = predict_item_item(user_id, movie_id, user_item_matrix, item_similarity, k=k)
        if not pd.isna(pred):
            predictions.append((movie_id, pred))

    recs = pd.DataFrame(predictions, columns=['movie_id', 'predicted_rating'])
    recs = recs.sort_values(by='predicted_rating', ascending=False).head(top_n)
    recs = recs.merge(movies_df[['movie_id', 'title']], on='movie_id', how='left')

    return recs[['movie_id', 'title', 'predicted_rating']]


def predict_on_test_set(test_df, user_item_matrix, user_similarity, item_similarity):
    user_user_preds = []
    item_item_preds = []

    for _, row in test_df.iterrows():
        user_id = row['user_id']
        movie_id = row['movie_id']
        true_rating = row['rating']

        pred_uu = predict_user_user(user_id, movie_id, user_item_matrix, user_similarity)
        pred_ii = predict_item_item(user_id, movie_id, user_item_matrix, item_similarity)

        user_user_preds.append([user_id, movie_id, true_rating, pred_uu])
        item_item_preds.append([user_id, movie_id, true_rating, pred_ii])

    uu_df = pd.DataFrame(user_user_preds, columns=['user_id', 'movie_id', 'true_rating', 'predicted_rating'])
    ii_df = pd.DataFrame(item_item_preds, columns=['user_id', 'movie_id', 'true_rating', 'predicted_rating'])

    return uu_df, ii_df


def save_outputs(user_user_preds, item_item_preds, out_dir='outputs'):
    os.makedirs(out_dir, exist_ok=True)

    user_user_preds.to_csv(os.path.join(out_dir, 'user_user_predictions.csv'), index=False)
    item_item_preds.to_csv(os.path.join(out_dir, 'item_item_predictions.csv'), index=False)


if __name__ == "__main__":
    ratings, movies, users = load_all()

    train, test = time_based_split(ratings, test_size=0.2)

    user_item_matrix = build_user_item_matrix(train)

    print("Computing user-user similarity...")
    user_similarity = compute_user_similarity(user_item_matrix)

    print("Computing item-item similarity...")
    item_similarity = compute_item_similarity(user_item_matrix)

    print("Predicting on test set...")
    uu_preds, ii_preds = predict_on_test_set(test, user_item_matrix, user_similarity, item_similarity)

    print("Saving outputs...")
    save_outputs(uu_preds, ii_preds)

    sample_user = user_item_matrix.index[0]

    print(f"\nTop recommendations for user {sample_user} using user-user CF:")
    print(recommend_user_user(sample_user, user_item_matrix, user_similarity, movies, top_n=10))

    print(f"\nTop recommendations for user {sample_user} using item-item CF:")
    print(recommend_item_item(sample_user, user_item_matrix, item_similarity, movies, top_n=10))