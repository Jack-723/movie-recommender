import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pickle
import pandas as pd
import streamlit as st

from collaborative_filtering import (
    recommend_user_user, recommend_item_item, recommend_svd
)
from content_based import recommend_bow_for_user, recommend_tfidf_for_user
from content_based_bert import recommend_bert_for_user
from nonpersonalized import recommend_popular, recommend_random, recommend_demographic

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')


@st.cache_resource
def load_models():
    def load(name):
        with open(os.path.join(MODELS_DIR, name), 'rb') as f:
            return pickle.load(f)

    return {
        'train': load('train.pkl'),
        'movies': load('movies.pkl'),
        'users': load('users.pkl'),
        'user_item_matrix': load('user_item_matrix.pkl'),
        'user_similarity': load('user_similarity.pkl'),
        'item_similarity': load('item_similarity.pkl'),
        'svd_model': load('svd_model.pkl'),
        'bow_sim': load('bow_sim.pkl'),
        'tfidf_sim': load('tfidf_sim.pkl'),
        'bert_sim': load('bert_sim.pkl'),
    }


def get_recommendations(user_id, model_name, data, n=10):
    train = data['train']
    movies = data['movies']
    users = data['users']

    if model_name == 'SVD':
        recs_df = recommend_svd(user_id, data['svd_model'], train, movies, n=n)
        return recs_df[['title', 'predicted_rating']].rename(columns={'predicted_rating': 'score'})

    elif model_name == 'User-User CF':
        recs_df = recommend_user_user(user_id, data['user_item_matrix'], data['user_similarity'], movies, top_n=n)
        return recs_df[['title', 'predicted_rating']].rename(columns={'predicted_rating': 'score'})

    elif model_name == 'Item-Item CF':
        recs_df = recommend_item_item(user_id, data['user_item_matrix'], data['item_similarity'], movies, top_n=n)
        return recs_df[['title', 'predicted_rating']].rename(columns={'predicted_rating': 'score'})

    elif model_name == 'Content-Based (BoW)':
        movie_ids = recommend_bow_for_user(user_id, train, movies, data['bow_sim'], n=n)
        titles = movies[movies['movie_id'].isin(movie_ids)][['movie_id', 'title']]
        titles = titles.set_index('movie_id').loc[movie_ids].reset_index()
        return titles[['title']]

    elif model_name == 'Content-Based (TF-IDF)':
        movie_ids = recommend_tfidf_for_user(user_id, train, movies, data['tfidf_sim'], n=n)
        titles = movies[movies['movie_id'].isin(movie_ids)][['movie_id', 'title']]
        titles = titles.set_index('movie_id').loc[movie_ids].reset_index()
        return titles[['title']]

    elif model_name == 'Content-Based (BERT)':
        movie_ids = recommend_bert_for_user(user_id, train, movies, data['bert_sim'], n=n)
        titles = movies[movies['movie_id'].isin(movie_ids)][['movie_id', 'title']]
        titles = titles.set_index('movie_id').loc[movie_ids].reset_index()
        return titles[['title']]

    elif model_name == 'Popularity':
        movie_ids = recommend_popular(user_id, train, movies, n=n)
        titles = movies[movies['movie_id'].isin(movie_ids)][['title']]
        return titles

    elif model_name == 'Random':
        movie_ids = recommend_random(user_id, train, movies, n=n)
        titles = movies[movies['movie_id'].isin(movie_ids)][['title']]
        return titles

    return pd.DataFrame()


def get_user_info(user_id, users):
    row = users[users['user_id'] == user_id]
    if row.empty:
        return None

    age_map = {
        1: 'under 18', 18: '18-24', 25: '25-34',
        35: '35-44', 45: '45-49', 50: '50-55', 56: '56+'
    }
    occupation_map = {
        0: 'other', 1: 'academic/educator', 2: 'artist', 3: 'clerical/admin',
        4: 'college/grad student', 5: 'customer service', 6: 'doctor/health care',
        7: 'executive/managerial', 8: 'farmer', 9: 'homemaker', 10: 'K-12 student',
        11: 'lawyer', 12: 'programmer', 13: 'retired', 14: 'sales/marketing',
        15: 'scientist', 16: 'self-employed', 17: 'technician/engineer',
        18: 'tradesman/craftsman', 19: 'unemployed', 20: 'writer'
    }

    info = row.iloc[0]
    return {
        'gender': 'Male' if info['gender'] == 'M' else 'Female',
        'age': age_map.get(info['age'], str(info['age'])),
        'occupation': occupation_map.get(info['occupation'], str(info['occupation']))
    }


# page config
st.set_page_config(
    page_title='Movie Recommender',
    page_icon='🎬',
    layout='centered'
)

# minimal custom styling
st.markdown("""
<style>
    .main { max-width: 800px; }
    .stSelectbox label { font-weight: 500; }
.user-card {
    background: #2b2b2b;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 20px;
    border: 1px solid #444;
    color: #ffffff;
}
.rec-item {
    padding: 10px 14px;
    border-radius: 8px;
    margin-bottom: 8px;
    background: #2b2b2b;
    border: 1px solid #444;
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

st.title("Movie Recommender")
st.caption("MovieLens 1M — Movie Recommendation Models")

# load models
with st.spinner("loading models..."):
    data = load_models()

st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    user_id = st.number_input(
        "User ID",
        min_value=1,
        max_value=6040,
        value=1,
        step=1
    )

with col2:
    model_name = st.selectbox(
        "Model",
        options=[
            'SVD',
            'Item-Item CF',
            'User-User CF',
            'Content-Based (BoW)',
            'Content-Based (TF-IDF)',
            'Content-Based (BERT)',
            'Popularity',
            'Random',
        ]
    )

# user info
user_info = get_user_info(user_id, data['users'])
if user_info:
    st.markdown(f"""
    <div class="user-card">
        <b>User {user_id}</b> &nbsp;·&nbsp;
        {user_info['gender']} &nbsp;·&nbsp;
        {user_info['age']} &nbsp;·&nbsp;
        {user_info['occupation']}
    </div>
    """, unsafe_allow_html=True)

# get recommendations
if st.button("Get Recommendations", type="primary", use_container_width=True):
    with st.spinner(f"getting recommendations from {model_name}..."):
        recs = get_recommendations(user_id, model_name, data, n=10)

    if recs is None or recs.empty:
        st.warning("no recommendations found for this user with this model.")
    else:
        # merge genres in
        recs = recs.merge(data['movies'][['title', 'genres']], on='title', how='left')

    st.subheader(f"Top 10 — {model_name}")
    for i, row in enumerate(recs.itertuples(), 1):
        score_str = f"{row.score:.2f}" if hasattr(row, 'score') and pd.notna(row.score) else ""
        genres = row.genres.replace('|', ', ') if pd.notna(row.genres) else ''
        st.markdown(f"""
        <div class="rec-item">
            <span><b>{i}.</b> {row.title} <span style="color: #aaa; font-size: 0.85em;">({genres})</span></span>
            <span style="color: #aaa; font-size: 0.9em;">{score_str}</span>
        </div>
        """, unsafe_allow_html=True)

    # show movies this user already rated highly (once, after all recs)
    with st.expander("movies this user rated highly"):
        liked = data['train'][
            (data['train']['user_id'] == user_id) &
            (data['train']['rating'] >= 4)
        ].merge(data['movies'][['movie_id', 'title', 'genres']], on='movie_id')
        liked['genres'] = liked['genres'].str.replace('|', ', ', regex=False)
        liked = liked[['title', 'genres', 'rating']].sort_values('rating', ascending=False).head(10)
        st.dataframe(liked, use_container_width=True, hide_index=True)