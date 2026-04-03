import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils import load_all
from nonpersonalized import time_based_split, evaluate_recommender
from collaborative_filtering import (
    build_user_item_matrix,
    compute_user_similarity,
    compute_item_similarity,
    predict_user_user,
    predict_item_item,
)
from content_based import build_corpus, build_bow_matrix, compute_bow_similarity


# ── weighted hybrid ─────────────────────────────────────────────────
# combines CF predicted ratings with content-based similarity scores
# using a simple weighted average: score = w * cf_score + (1-w) * cb_score

def predict_weighted_hybrid(user_id, movie_id, user_item_matrix, item_similarity,
                            cb_sim_df, train, w_cf=0.7, k=10):
    """
    Weighted blend of item-item CF and content-based similarity.
    w_cf: weight for CF component (content gets 1 - w_cf)
    """
    # CF prediction
    cf_pred = predict_item_item(user_id, movie_id, user_item_matrix, item_similarity, k=k)

    # content-based score: average similarity to user's liked movies
    cb_pred = _content_score_for_user_movie(user_id, movie_id, cb_sim_df, train)

    if pd.isna(cf_pred) and pd.isna(cb_pred):
        return np.nan
    elif pd.isna(cf_pred):
        return cb_pred  # fallback to content if CF can't predict
    elif pd.isna(cb_pred):
        return cf_pred  # fallback to CF if content can't predict
    else:
        return w_cf * cf_pred + (1 - w_cf) * cb_pred


def _content_score_for_user_movie(user_id, movie_id, cb_sim_df, train):
    """
    Estimate a rating-like score from content similarity.
    Take the user's highly rated movies, compute avg similarity to target movie,
    then scale to 1-5 range.
    """
    if movie_id not in cb_sim_df.index:
        return np.nan

    user_ratings = train[train['user_id'] == user_id]
    liked = user_ratings[user_ratings['rating'] >= 4]['movie_id']
    liked = [m for m in liked if m in cb_sim_df.index and m != movie_id]

    if not liked:
        return np.nan

    # average content similarity between this movie and all liked movies
    avg_sim = cb_sim_df.loc[liked, movie_id].mean()

    # scale similarity (0-1) to rating range (1-5)
    scaled = 1 + avg_sim * 4
    return scaled


def recommend_weighted_hybrid(user_id, train, movies, user_item_matrix,
                              item_similarity, cb_sim_df, n=10, w_cf=0.7, k=10):
    """Top-N recommendations using the weighted hybrid approach."""
    if user_id not in user_item_matrix.index:
        return []

    seen = set(train[train['user_id'] == user_id]['movie_id'])
    all_movies = set(movies['movie_id'])
    candidates = list(all_movies - seen)

    scores = []
    for mid in candidates:
        s = predict_weighted_hybrid(
            user_id, mid, user_item_matrix, item_similarity,
            cb_sim_df, train, w_cf=w_cf, k=k
        )
        if not pd.isna(s):
            scores.append((mid, s))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [m for m, _ in scores[:n]]


# ── switching hybrid ────────────────────────────────────────────────
# idea: use CF when we have enough data for the user, fall back to
# content-based for cold-start users (few ratings in training set)

# Note: MovieLens 1M has no true cold-start users (all have 20+ ratings).
# The switching hybrid's fallback logic is correct but won't trigger often in evaluation.
# In production with real new users, the content-based fallback would activate.
COLD_START_THRESHOLD = 5  # if user has fewer ratings than this, they're "cold"


def recommend_switching_hybrid(user_id, train, movies, user_item_matrix,
                               item_similarity, cb_sim_df, n=10,
                               cold_threshold=COLD_START_THRESHOLD, k=10):
    """
    Switching hybrid: CF for warm users, content-based for cold-start users.
    """
    user_ratings = train[train['user_id'] == user_id]
    n_ratings = len(user_ratings)

    if n_ratings < cold_threshold:
        # cold-start → use content-based
        return _content_based_recs(user_id, train, movies, cb_sim_df, n=n)
    else:
        # warm user → use item-item CF
        return _cf_recs(user_id, train, movies, user_item_matrix, item_similarity, n=n, k=k)


def _cf_recs(user_id, train, movies, user_item_matrix, item_similarity, n=10, k=10):
    """Get top-N recs using item-item CF."""
    if user_id not in user_item_matrix.index:
        return []

    user_ratings = user_item_matrix.loc[user_id]
    unseen = user_ratings[user_ratings.isna()].index

    preds = []
    for mid in unseen:
        p = predict_item_item(user_id, mid, user_item_matrix, item_similarity, k=k)
        if not pd.isna(p):
            preds.append((mid, p))

    preds.sort(key=lambda x: x[1], reverse=True)
    return [m for m, _ in preds[:n]]


def _content_based_recs(user_id, train, movies, cb_sim_df, n=10):
    """Content-based recs for cold-start users."""
    liked = train[(train['user_id'] == user_id) & (train['rating'] >= 4)]['movie_id']
    liked = [m for m in liked if m in cb_sim_df.index]

    if not liked:
        # absolute cold start (no likes at all) → return popular movies
        popular = train.groupby('movie_id')['rating'].mean().sort_values(ascending=False)
        seen = set(train[train['user_id'] == user_id]['movie_id'])
        return [m for m in popular.index if m not in seen][:n]

    seen = set(train[train['user_id'] == user_id]['movie_id'])
    scores = cb_sim_df.loc[liked].mean(axis=0)
    scores = scores.drop(index=[m for m in liked if m in scores.index])
    scores = scores[~scores.index.isin(seen)]

    return list(scores.sort_values(ascending=False).head(n).index)


# ── evaluation helpers ──────────────────────────────────────────────

def evaluate_hybrid(recommend_fn, train, test, movies, n=10, sample_size=500):
    """
    Same evaluation logic as nonpersonalized.evaluate_recommender but
    the recommend_fn here already has the models baked in via closure.
    """
    from nonpersonalized import precision_at_k, recall_at_k, hit_rate, ndcg_at_k

    test_liked = test[test['rating'] >= 4]
    eligible = test_liked['user_id'].unique()

    if len(eligible) > sample_size:
        eval_users = np.random.choice(eligible, size=sample_size, replace=False)
    else:
        eval_users = eligible

    precisions, recalls, hits, ndcgs = [], [], [], []

    for uid in eval_users:
        relevant = set(test_liked[test_liked['user_id'] == uid]['movie_id'])
        recs = recommend_fn(uid, train, movies, n=n)

        precisions.append(precision_at_k(recs, relevant))
        recalls.append(recall_at_k(recs, relevant))
        hits.append(hit_rate(recs, relevant))
        ndcgs.append(ndcg_at_k(recs, relevant))

    return {
        'precision@10': np.mean(precisions),
        'recall@10': np.mean(recalls),
        'hit_rate': np.mean(hits),
        'ndcg@10': np.mean(ndcgs),
    }


# ── main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ratings, movies, users = load_all()
    train, test = time_based_split(ratings, test_size=0.2)

    # build models needed by the hybrids
    print("building user-item matrix...")
    user_item_matrix = build_user_item_matrix(train)

    print("computing item-item similarity (CF)...")
    item_similarity = compute_item_similarity(user_item_matrix)

    print("building content-based similarity (BoW)...")
    corpus = build_corpus(movies)
    bow_matrix, vectorizer = build_bow_matrix(corpus)
    cb_sim_df = compute_bow_similarity(bow_matrix, movies)

    np.random.seed(42)

    # evaluate weighted hybrid
    print("\nevaluating weighted hybrid (w_cf=0.7)...")

    def weighted_fn(uid, train, movies, n=10):
        return recommend_weighted_hybrid(
            uid, train, movies, user_item_matrix, item_similarity,
            cb_sim_df, n=n, w_cf=0.7
        )

    results_weighted = evaluate_hybrid(weighted_fn, train, test, movies, n=10)
    print(f"weighted hybrid: {results_weighted}")

    # evaluate switching hybrid
    print("\nevaluating switching hybrid...")

    def switching_fn(uid, train, movies, n=10):
        return recommend_switching_hybrid(
            uid, train, movies, user_item_matrix, item_similarity,
            cb_sim_df, n=n
        )

    results_switching = evaluate_hybrid(switching_fn, train, test, movies, n=10)
    print(f"switching hybrid: {results_switching}")
