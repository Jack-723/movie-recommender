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


def evaluate_regression(preds_df, label=''):
    # drop rows where prediction is NaN (cold-start cases)
    valid = preds_df.dropna(subset=['predicted_rating'])
    mae = np.mean(np.abs(valid['true_rating'] - valid['predicted_rating']))
    rmse = np.sqrt(np.mean((valid['true_rating'] - valid['predicted_rating']) ** 2))
    print(f"{label} — MAE: {mae:.4f}, RMSE: {rmse:.4f} (on {len(valid)} predictions)")
    return {'mae': mae, 'rmse': rmse}


def save_outputs(user_user_preds, item_item_preds, out_dir='outputs'):
    os.makedirs(out_dir, exist_ok=True)

    user_user_preds.to_csv(os.path.join(out_dir, 'user_user_predictions.csv'), index=False)
    item_item_preds.to_csv(os.path.join(out_dir, 'item_item_predictions.csv'), index=False)


# model-based CF using SVD (matrix factorization implemented with numpy)
# learns latent user and item vectors by minimizing prediction error via SGD

class SVDModel:
    def __init__(self, n_factors=50, n_epochs=20, lr=0.005, reg=0.02):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg

    def fit(self, train):
        # build index maps so we can work with integer indices internally
        self.user_ids = list(train['user_id'].unique())
        self.movie_ids = list(train['movie_id'].unique())
        self.user_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.movie_idx = {m: i for i, m in enumerate(self.movie_ids)}

        n_users = len(self.user_ids)
        n_movies = len(self.movie_ids)

        self.global_mean = train['rating'].mean()

        # initialize latent factors and biases randomly
        self.P = np.random.normal(0, 0.1, (n_users, self.n_factors))   # user factors
        self.Q = np.random.normal(0, 0.1, (n_movies, self.n_factors))  # item factors
        self.bu = np.zeros(n_users)   # user biases
        self.bi = np.zeros(n_movies)  # item biases

        # SGD over all ratings for n_epochs
        for epoch in range(self.n_epochs):
            for _, row in train.iterrows():
                u = self.user_idx.get(row['user_id'])
                m = self.movie_idx.get(row['movie_id'])
                if u is None or m is None:
                    continue

                pred = self.global_mean + self.bu[u] + self.bi[m] + self.P[u] @ self.Q[m]
                err = row['rating'] - pred

                # update biases and factors
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[m] += self.lr * (err - self.reg * self.bi[m])
                self.P[u] += self.lr * (err * self.Q[m] - self.reg * self.P[u])
                self.Q[m] += self.lr * (err * self.P[u] - self.reg * self.Q[m])

            if (epoch + 1) % 5 == 0:
                print(f"  epoch {epoch + 1}/{self.n_epochs} done")

        return self

    def predict(self, user_id, movie_id):
        u = self.user_idx.get(user_id)
        m = self.movie_idx.get(movie_id)

        if u is None or m is None:
            return self.global_mean  # fallback for unseen users/items

        return self.global_mean + self.bu[u] + self.bi[m] + self.P[u] @ self.Q[m]


def train_svd(train, n_factors=50, n_epochs=20, lr=0.005, reg=0.02):
    model = SVDModel(n_factors=n_factors, n_epochs=n_epochs, lr=lr, reg=reg)
    model.fit(train)
    return model


def evaluate_svd(model, test):
    preds = []
    for _, row in test.iterrows():
        pred = model.predict(row['user_id'], row['movie_id'])
        preds.append({
            'user_id': row['user_id'],
            'movie_id': row['movie_id'],
            'true_rating': row['rating'],
            'predicted_rating': pred
        })

    preds_df = pd.DataFrame(preds)
    return evaluate_regression(preds_df, label='SVD')


def recommend_svd(user_id, model, train, movies, n=10):
    # predict ratings for all movies this user hasn't seen
    seen = set(train[train['user_id'] == user_id]['movie_id'])
    all_movies = movies['movie_id'].tolist()
    unseen = [m for m in all_movies if m not in seen]

    preds = [(mid, model.predict(user_id, mid)) for mid in unseen]
    preds.sort(key=lambda x: x[1], reverse=True)

    recs = pd.DataFrame(preds[:n], columns=['movie_id', 'predicted_rating'])
    recs = recs.merge(movies[['movie_id', 'title']], on='movie_id', how='left')
    return recs[['movie_id', 'title', 'predicted_rating']]


def recommend_svd_list(user_id, model, train, movies, n=10):
    # same as above but returns a plain list for evaluation
    seen = set(train[train['user_id'] == user_id]['movie_id'])
    all_movies = movies['movie_id'].tolist()
    unseen = [m for m in all_movies if m not in seen]

    preds = [(mid, model.predict(user_id, mid)) for mid in unseen]
    preds.sort(key=lambda x: x[1], reverse=True)
    return [mid for mid, _ in preds[:n]]


if __name__ == "__main__":
    ratings, movies, users = load_all()

    train, test = time_based_split(ratings, test_size=0.2)

    user_item_matrix = build_user_item_matrix(train)

    print("computing user-user similarity...")
    user_similarity = compute_user_similarity(user_item_matrix)

    print("computing item-item similarity...")
    item_similarity = compute_item_similarity(user_item_matrix)

    print("predicting on test set...")
    uu_preds, ii_preds = predict_on_test_set(test, user_item_matrix, user_similarity, item_similarity)

    print("\nevaluating regression metrics...")
    evaluate_regression(uu_preds, label='user-user CF')
    evaluate_regression(ii_preds, label='item-item CF')

    print("\nsaving outputs...")
    save_outputs(uu_preds, ii_preds)

    sample_user = user_item_matrix.index[0]

    print(f"\ntop recommendations for user {sample_user} using user-user CF:")
    print(recommend_user_user(sample_user, user_item_matrix, user_similarity, movies, top_n=10))

    print(f"\ntop recommendations for user {sample_user} using item-item CF:")
    print(recommend_item_item(sample_user, user_item_matrix, item_similarity, movies, top_n=10))

    # SVD
    print("\ntraining SVD model...")
    svd_model = train_svd(train)

    print("\nevaluating SVD...")
    evaluate_svd(svd_model, test)

    print(f"\ntop recommendations for user {sample_user} using SVD:")
    print(recommend_svd(sample_user, svd_model, train, movies, n=10))