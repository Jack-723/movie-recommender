import sys
import os
sys.path.append(os.path.dirname(__file__))

# Load data from utils module
from utils import load_all
import pandas as pd


def inspect_structure(ratings, movies, users):
    # Display basic info about each dataset (dimensions, first few rows, data types)
    print("\nRatings:")
    print(f"Shape: {ratings.shape}")
    print(ratings.head())
    print(ratings.dtypes)

    print("\nMovies:")
    print(f"Shape: {movies.shape}")
    print(movies.head())
    print(movies.dtypes)

    print("\nUsers:")
    print(f"Shape: {users.shape}")
    print(users.head())
    print(users.dtypes)


def check_missing(ratings, movies, users):
    # Count null values in each dataset to identify missing data
    print("\nMssing values:")
    print("ratings:", ratings.isnull().sum().to_dict())
    print("movies: ", movies.isnull().sum().to_dict())
    print("users:  ", users.isnull().sum().to_dict())


def check_duplicates(ratings, movies, users):
    print("\nDuplicates:")

    # Check for duplicate ratings (same user rating same movie multiple times)
    rating_dups = ratings.duplicated(subset=['user_id', 'movie_id']).sum()
    print(f"duplicate ratings: {rating_dups}")

    movie_dups = movies.duplicated(subset=['movie_id']).sum()
    print(f"duplicate movies: {movie_dups}")

    user_dups = users.duplicated(subset=['user_id']).sum()
    print(f"duplicate users: {user_dups}")


def basic_stats(ratings, movies, users):
    # Calculate and display overall statistics about the rating data
    print("\nBasic Stats:")
    print(f"total ratings:   {len(ratings)}")
    print(f"unique users:    {ratings['user_id'].nunique()}")
    print(f"unique movies:   {ratings['movie_id'].nunique()}")
    print(f"rating range:    {ratings['rating'].min()} to {ratings['rating'].max()}")
    print(f"avg rating:      {ratings['rating'].mean():.2f}")

    # Calculate sparsity: how many potential ratings are missing from the matrix
    n_users = ratings['user_id'].nunique()
    n_movies = ratings['movie_id'].nunique()
    sparsity = 1 - len(ratings) / (n_users * n_movies)
    print(f"matrix sparsity: {sparsity:.4f} ({sparsity*100:.2f}% empty)")


if __name__ == "__main__":
    ratings, movies, users = load_all()

    inspect_structure(ratings, movies, users)
    check_missing(ratings, movies, users)
    check_duplicates(ratings, movies, users)
    basic_stats(ratings, movies, users)

    print("\ndone")