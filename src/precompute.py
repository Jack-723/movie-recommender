import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pickle
import numpy as np

from utils import load_all
from nonpersonalized import time_based_split
from collaborative_filtering import (
    build_user_item_matrix, compute_user_similarity,
    compute_item_similarity, train_svd
)
from content_based import (
    build_corpus, build_bow_matrix, compute_bow_similarity,
    build_tfidf_matrix, compute_tfidf_similarity
)
from content_based_bert import build_bert_corpus, compute_bert_embeddings, compute_bert_similarity

# everything gets saved here
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def save(obj, name):
    path = os.path.join(MODELS_DIR, name)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"saved {name}")


if __name__ == "__main__":
    ratings, movies, users = load_all()
    train, test = time_based_split(ratings, test_size=0.2)

    # save train/test and base data so the app can use them
    save(train, 'train.pkl')
    save(test, 'test.pkl')
    save(movies, 'movies.pkl')
    save(users, 'users.pkl')

    # user-item matrix
    print("building user-item matrix...")
    user_item_matrix = build_user_item_matrix(train)
    save(user_item_matrix, 'user_item_matrix.pkl')

    # user-user similarity
    print("computing user-user similarity...")
    user_similarity = compute_user_similarity(user_item_matrix)
    save(user_similarity, 'user_similarity.pkl')

    # item-item similarity
    print("computing item-item similarity...")
    item_similarity = compute_item_similarity(user_item_matrix)
    save(item_similarity, 'item_similarity.pkl')

    # SVD
    print("training SVD...")
    svd_model = train_svd(train)
    save(svd_model, 'svd_model.pkl')

    # BoW
    print("building BoW similarity...")
    corpus = build_corpus(movies)
    bow_matrix, bow_vectorizer = build_bow_matrix(corpus)
    bow_sim = compute_bow_similarity(bow_matrix, movies)
    save(bow_sim, 'bow_sim.pkl')
    save(corpus, 'corpus.pkl')

    # TF-IDF
    print("building TF-IDF similarity...")
    tfidf_matrix, tfidf_vectorizer = build_tfidf_matrix(corpus)
    tfidf_sim = compute_tfidf_similarity(tfidf_matrix, movies)
    save(tfidf_sim, 'tfidf_sim.pkl')

    # BERT
    print("building BERT similarity (this will take a few minutes)...")
    bert_corpus = build_bert_corpus(movies)
    bert_embeddings = compute_bert_embeddings(bert_corpus)
    bert_sim = compute_bert_similarity(bert_embeddings, movies)
    save(bert_sim, 'bert_sim.pkl')

    print("\nall models saved to models/")
    print("you can now run the app with: streamlit run src/app.py")