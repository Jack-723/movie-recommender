import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_all
import pandas as pd
import numpy as np 



def time_based_split(ratings, test_size= 0.2):

    ratings_sorted = ratings.sort_values(['user_id', 'timestamp'])
    train_list = []
    test_list = []



    for user_id, user_ratings in ratings_sorted.groupby('user_id'): #Sort all ratings by user, then by timestamp within each user.
                                                                    # So user 1's oldest rating comes first, newest last.
        n = len(user_ratings) # how many ratings this user has.
        cutoff = int(n*(1-test_size))
        train_list.append(user_ratings.iloc[:cutoff])
        test_list.append(user_ratings.iloc[cutoff:]) # cuttoff 80 percent of data for train, with 50 ratings, cutoff = 40. So the first 40 (oldest) go to train


    train = pd.concat(train_list, ignore_index=True)
    test = pd.concat(test_list, ignore_index=True)

    print(f"train: {len(train)} ratings, test: {len(test)} ratings")
    return train, test





def recommend_random(user_id, train, movies, n= 10) : 
    seen = set(train[train['user_id'] == user_id]['movie_id'])

    all_movies = set(movies['movie_id'])
    unseen = list(all_movies - seen)

    recs = np.random.choice(unseen, size = min(n, len(unseen)), replace=False)
    return recs



def recommend_popular(user_id ,train, movies, n=10, min_ratings = 50) : 
    # compute popularity from training data only

    movie_stats = train.groupby('movie_id')['rating'].agg(['count', 'mean'])
    movie_stats.columns = ['num_ratings', 'avg_rating']


    # filter out movies with too few ratings
    popular = movie_stats[movie_stats['num_ratings'] >= min_ratings] 
    

    # sort by average rating, break ties by number of ratings
    popular = popular.sort_values(['avg_rating', 'num_ratings'], ascending=False)


    seen = set(train[train['user_id'] == user_id]['movie_id'])
    recs = [m for m in popular.index if m not in seen][:n]
    return recs      # exclude movies this user already rated

def recommend_demographic(user_id, train, movies, users, group_by = 'gender', n=10, min_ratings = 20): 
    user_info = users[users['user_id'] == user_id].iloc[0]    # get this user's demographic info

    age_map = {    # map age codes to labels if grouping by age

        1: 'Under 18',
        18: '18-24',
        25: '25-34',
        35: '35-44',
        45: '45-49',
        50: '50-55',
        56: '56+'
    }

    if group_by == 'gender' :     # figure out which group this user belongs to
        group_value = user_info['gender']
    elif group_by == 'age':
        group_value = age_map[user_info['age']]
        users = users.copy()
        users['age_group'] = users['age'].map(age_map)
        group_by = 'age_group'
    elif group_by == 'occupation':
        group_value = user_info['occupation']
    else:
        raise ValueError(f"unknown group_by: {group_by}")
    

    
    if group_by == 'age_group' :    # get all users in the same group
        group_users = users[users['age_group'] == group_value]['user_id']
    else:
        group_users = users[users[group_by] == group_value]['user_id']



    group_ratings = train[train['user_id'].isin(group_users)]    # filter training ratings to only this demographic group

    # same popularity logic but within the group
    movie_stats = group_ratings.groupby('movie_id')['rating'].agg(['count', 'mean'])
    movie_stats.columns = ['num_ratings', 'avg_rating'] 

    popular = movie_stats[movie_stats['num_ratings'] >= min_ratings]
    popular = popular.sort_values(['avg_rating', 'num_ratings'], ascending=False)

    seen = set(train[train['user_id'] == user_id]['movie_id'])    # exclude movies this user already rated
    recs = [int(m) for m in popular.index if m not in seen][:n]

    return recs


def precision_at_k(recommended, relevant) :    # of the K recommended movies, how many were relevant?
    if len(recommended) == 0 : 
        return 0.0
    hits = len(set(recommended)& set(relevant))
    return hits / len(recommended)

    # of all relevant movies, how many did we recommend?
def recall_at_k(recommended, relevant) :
    if len(relevant) == 0 : 
        return 0.0
    hits = len(set(recommended) & set(relevant))
    return hits / len(relevant)

def hit_rate(recommended, relevant) :
    return 1.0 if len(set(recommended) & set(relevant)) > 0 else 0.0    # did at least one recommendation land?



def evaluate_recommender(recommend_fn, train, test, movies, users=None, n=10, sample_size=500):
    # get users who have at least 1 liked movie in test set
    test_liked = test[test['rating'] >= 4]
    eligible_users = test_liked['user_id'].unique()

    # sample users for speed (evaluating all 6040 is slow)
    if len(eligible_users) > sample_size:
        eval_users = np.random.choice(eligible_users, size=sample_size, replace=False)
    else:
        eval_users = eligible_users

    precisions = []
    recalls = []
    hits = []

    for uid in eval_users:
        # get this user's liked movies in test set (ground truth)
        relevant = set(test_liked[test_liked['user_id'] == uid]['movie_id'])

        # get recommendations
        if users is not None:
            recs = recommend_fn(uid, train, movies, users, n=n)
        else:
            recs = recommend_fn(uid, train, movies, n=n)

        precisions.append(precision_at_k(recs, relevant))
        recalls.append(recall_at_k(recs, relevant))
        hits.append(hit_rate(recs, relevant))

    return {
        'precision@10': np.mean(precisions),
        'recall@10': np.mean(recalls),
        'hit_rate': np.mean(hits)
    }




if __name__ == "__main__":
    ratings, movies,  users = load_all()
    train, test = time_based_split(ratings)

    np.random.seed(42)  # reproducible results

    print("\nevaluating recommenders...")

     # random baseline
    results_random = evaluate_recommender(recommend_random, train, test, movies, n=10)
    print(f"random:                {results_random}")

     # popularity
    results_popular = evaluate_recommender(recommend_popular, train, test, movies, n=10)
    print(f"popular:               {results_popular}")
     # demographic - need wrapper functions since evaluate_recommender expects a standard signature
    def demo_gender(uid, train, movies, users, n=10):
         return recommend_demographic(uid, train, movies, users, group_by='gender', n=n)
    def demo_age(uid, train, movies, users, n=10):
         return recommend_demographic(uid, train, movies, users, group_by='age', n=n)

    def demo_occupation(uid, train, movies, users, n=10):
         return recommend_demographic(uid, train, movies, users, group_by='occupation', n=n)

    results_gender = evaluate_recommender(demo_gender, train, test, movies, users, n=10)
    print(f"demographic (gender):  {results_gender}")

    results_age = evaluate_recommender(demo_age, train, test, movies, users, n=10)
    print(f"demographic (age):     {results_age}")

    results_occ = evaluate_recommender(demo_occupation, train, test, movies, users, n=10)
    print(f"demographic (occup):   {results_occ}")

    print("\ndone")