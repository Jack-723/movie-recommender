import pandas as pd
import os

# default path to the raw data folder
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')


def load_ratings(path=DATA_PATH):
    # ratings.dat format: UserID::MovieID::Rating::Timestamp
    filepath = os.path.join(path, 'ratings.dat')
    df = pd.read_csv(
        filepath,
        sep='::',
        engine='python',
        names=['user_id', 'movie_id', 'rating', 'timestamp']
    )
    return df


def load_movies(path=DATA_PATH):
    # movies.dat format: MovieID::Title::Genres
    filepath = os.path.join(path, 'movies.dat')
    df = pd.read_csv(
        filepath,
        sep='::',
        engine='python',
        names=['movie_id', 'title', 'genres'],
        encoding='latin-1'  # some movie titles have special characters
    )
    return df


def load_users(path=DATA_PATH):
    # users.dat format: UserID::Gender::Age::Occupation::Zip-code
    filepath = os.path.join(path, 'users.dat')
    df = pd.read_csv(
        filepath,
        sep='::',
        engine='python',
        names=['user_id', 'gender', 'age', 'occupation', 'zip_code']
    )
    return df


def load_all(path=DATA_PATH):
    print("loading data")
    ratings = load_ratings(path)
    movies = load_movies(path)
    users = load_users(path)
    print("done")
    return ratings, movies, users