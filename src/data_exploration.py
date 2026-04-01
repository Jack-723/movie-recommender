''' This file is for data exploration, it loads the MovieLens 1M dataset, inspects structure, checks for
missing values, duplicates, and does basic exploratory analysis
'''

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load data from utils module
from utils import load_all
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


def rating_distribution(ratings):
    # how often each rating value (1-5) appears
    print("\nRating distribution:")
    print(ratings['rating'].value_counts().sort_index())

    plt.figure(figsize=(7, 4))
    sns.countplot(x='rating', data=ratings)
    plt.title("rating distribution")
    plt.xlabel("rating")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig("outputs/figures/rating_distribution.png")
    plt.close()
    print("saved rating_distribution.png")


def user_activity(ratings):
    # how many ratings each user has given
    ratings_per_user = ratings.groupby('user_id')['rating'].count()

    print("\nRatings per user:")
    print(ratings_per_user.describe())

    plt.figure(figsize=(7, 4))
    ratings_per_user.hist(bins=50)
    plt.title("ratings per user")
    plt.xlabel("number of ratings")
    plt.ylabel("number of users")
    plt.tight_layout()
    plt.savefig("outputs/figures/ratings_per_user.png")
    plt.close()
    print("saved ratings_per_user.png")


def item_popularity(ratings, movies):
    # how many ratings each movie has received
    ratings_per_movie = ratings.groupby('movie_id')['rating'].count().reset_index()
    ratings_per_movie.columns = ['movie_id', 'num_ratings']
    ratings_per_movie = ratings_per_movie.merge(movies[['movie_id', 'title']], on='movie_id')
    ratings_per_movie = ratings_per_movie.sort_values('num_ratings', ascending=False)

    print("\nMost rated movies:")
    print(ratings_per_movie.head(10)[['title', 'num_ratings']].to_string(index=False))

    print("\nRatings per movie:")
    print(ratings_per_movie['num_ratings'].describe())

    plt.figure(figsize=(7, 4))
    ratings_per_movie['num_ratings'].hist(bins=50)
    plt.title("ratings per movie")
    plt.xlabel("number of ratings")
    plt.ylabel("number of movies")
    plt.tight_layout()
    plt.savefig("outputs/figures/ratings_per_movie.png")
    plt.close()
    print("saved ratings_per_movie.png")


def demographic_analysis(users, ratings):
    # age groups are stored as codes in the dataset, map them to readable labels
    age_map = {
        1: "under 18",
        18: "18-24",
        25: "25-34",
        35: "35-44",
        45: "45-49",
        50: "50-55",
        56: "56+"
    }
    users['age_group'] = users['age'].map(age_map)
 
    print("\nGender breakdown:")
    print(users['gender'].value_counts())
 
    print("\nAge group breakdown:")
    print(users['age_group'].value_counts().sort_index())
 
    print("\nTop occupations:")
    # occupation is stored as a code, the mapping is from the dataset readme
    occupation_map = {
        0: "other", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
        4: "college/grad student", 5: "customer service", 6: "doctor/health care",
        7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student",
        11: "lawyer", 12: "programmer", 13: "retired", 14: "sales/marketing",
        15: "scientist", 16: "self-employed", 17: "technician/engineer",
        18: "tradesman/craftsman", 19: "unemployed", 20: "writer"
    }
    users['occupation_label'] = users['occupation'].map(occupation_map)
    print(users['occupation_label'].value_counts().head(10))
 
    # average rating by gender
    merged = ratings.merge(users[['user_id', 'gender', 'age_group', 'occupation_label']], on='user_id')
    print("\nAverage rating by gender:")
    print(merged.groupby('gender')['rating'].mean().round(2))
 
    print("\nAverage rating by age group:")
    print(merged.groupby('age_group')['rating'].mean().round(2))
 
    # plot gender distribution
    plt.figure(figsize=(5, 4))
    users['gender'].value_counts().plot(kind='bar')
    plt.title("users by gender")
    plt.xlabel("gender")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig("outputs/figures/gender_distribution.png")
    plt.close()
    print("saved gender_distribution.png")
 
    # plot age group distribution
    plt.figure(figsize=(7, 4))
    users['age_group'].value_counts().sort_index().plot(kind='bar')
    plt.title("users by age group")
    plt.xlabel("age group")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig("outputs/figures/age_distribution.png")
    plt.close()
    print("saved age_distribution.png")


def genre_analysis(movies, ratings):
    # genres are pipe-separated so we need to split them out into individual rows
    genre_rows = movies['genres'].str.split('|').explode()
    genre_counts = genre_rows.value_counts()
 
    print("\nGenre distribution (number of movies per genre):")
    print(genre_counts)
 
    # average rating per genre
    # need to explode genres and merge with ratings
    movies_exploded = movies.copy()
    movies_exploded['genre'] = movies['genres'].str.split('|')
    movies_exploded = movies_exploded.explode('genre')
 
    merged = ratings.merge(movies_exploded[['movie_id', 'genre']], on='movie_id')
    avg_rating_by_genre = merged.groupby('genre')['rating'].mean().sort_values(ascending=False)
 
    print("\nAverage rating by genre:")
    print(avg_rating_by_genre.round(2))
 
    # plot number of movies per genre
    plt.figure(figsize=(10, 5))
    genre_counts.plot(kind='bar')
    plt.title("movies per genre")
    plt.xlabel("genre")
    plt.ylabel("number of movies")
    plt.tight_layout()
    plt.savefig("outputs/figures/genre_distribution.png")
    plt.close()
    print("saved genre_distribution.png")
 
    # plot average rating per genre
    plt.figure(figsize=(10, 5))
    avg_rating_by_genre.plot(kind='bar')
    plt.title("average rating by genre")
    plt.xlabel("genre")
    plt.ylabel("avg rating")
    plt.tight_layout()
    plt.savefig("outputs/figures/avg_rating_by_genre.png")
    plt.close()
    print("saved avg_rating_by_genre.png")


if __name__ == "__main__":
    ratings, movies, users = load_all()

    inspect_structure(ratings, movies, users)
    check_missing(ratings, movies, users)
    check_duplicates(ratings, movies, users)
    basic_stats(ratings, movies, users)
    rating_distribution(ratings)
    user_activity(ratings)
    item_popularity(ratings, movies)
    demographic_analysis(users, ratings)
    genre_analysis(movies, ratings)

    print("\ndone")