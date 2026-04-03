import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_all
from nonpersonalized import (
    time_based_split, recommend_random, recommend_popular,
    recommend_demographic, evaluate_recommender,
    precision_at_k, recall_at_k, hit_rate
)
from collaborative_filtering import (
    build_user_item_matrix, compute_item_similarity, compute_user_similarity,
    recommend_item_item, recommend_user_user
)
from content_based import (
    build_corpus, build_bow_matrix, compute_bow_similarity,
    recommend_bow_for_user
)
from hybrid_recommender import (
    recommend_weighted_hybrid, recommend_switching_hybrid,
    evaluate_hybrid
)


def run_full_evaluation(train, test, movies, users,
                        user_item_matrix, item_similarity, user_similarity, cb_sim_df,
                        n=10, sample_size=200):
    """Evaluate every model on the same test set and return a results dict."""

    np.random.seed(42)
    results = {}

    # -- non-personalized --
    print("evaluating random...")
    results['Random'] = evaluate_recommender(recommend_random, train, test, movies, n=n, sample_size=sample_size)

    print("evaluating popularity...")
    results['Popularity'] = evaluate_recommender(recommend_popular, train, test, movies, n=n, sample_size=sample_size)

    print("evaluating demographic (gender)...")
    def demo_gender(uid, train, movies, users, n=10):
        return recommend_demographic(uid, train, movies, users, group_by='gender', n=n)
    results['Demographic (gender)'] = evaluate_recommender(demo_gender, train, test, movies, users, n=n, sample_size=sample_size)

    print("evaluating demographic (age)...")
    def demo_age(uid, train, movies, users, n=10):
        return recommend_demographic(uid, train, movies, users, group_by='age', n=n)
    results['Demographic (age)'] = evaluate_recommender(demo_age, train, test, movies, users, n=n, sample_size=sample_size)

    # -- collaborative filtering --
    print("evaluating user-user CF...")
    def uu_cf_fn(uid, train, movies, n=10):
        recs_df = recommend_user_user(uid, user_item_matrix, user_similarity, movies, top_n=n, k=10)
        return list(recs_df['movie_id'])
    results['User-User CF'] = evaluate_recommender(uu_cf_fn, train, test, movies, n=n, sample_size=sample_size)

    print("evaluating item-item CF...")
    def ii_cf_fn(uid, train, movies, n=10):
        recs_df = recommend_item_item(uid, user_item_matrix, item_similarity, movies, top_n=n, k=10)
        return list(recs_df['movie_id'])
    results['Item-Item CF'] = evaluate_recommender(ii_cf_fn, train, test, movies, n=n, sample_size=sample_size)

    # -- content-based --
    print("evaluating BoW content-based...")
    def bow_fn(uid, train, movies, n=10):
        return recommend_bow_for_user(uid, train, movies, cb_sim_df, n=n)
    results['Content-Based (BoW)'] = evaluate_recommender(bow_fn, train, test, movies, n=n, sample_size=sample_size)

    # -- hybrids --
    print("evaluating weighted hybrid (w=0.7)...")
    def weighted_fn(uid, train, movies, n=10):
        return recommend_weighted_hybrid(uid, train, movies, user_item_matrix, item_similarity, cb_sim_df, n=n, w_cf=0.7)
    results['Weighted Hybrid'] = evaluate_hybrid(weighted_fn, train, test, movies, n=n, sample_size=sample_size)

    print("evaluating switching hybrid...")
    def switching_fn(uid, train, movies, n=10):
        return recommend_switching_hybrid(uid, train, movies, user_item_matrix, item_similarity, cb_sim_df, n=n)
    results['Switching Hybrid'] = evaluate_hybrid(switching_fn, train, test, movies, n=n, sample_size=sample_size)

    return results


def build_comparison_table(results):
    """Turn the results dict into a sorted DataFrame."""
    df = pd.DataFrame(results).T
    df.index.name = 'Model'
    df = df.sort_values('precision@10', ascending=False)
    return df


def compute_lift(results_df):
    """Compute lift of each model over the random baseline."""
    random_row = results_df.loc['Random']
    lift = results_df.copy()
    for col in results_df.columns:
        if random_row[col] > 0:
            lift[col] = results_df[col] / random_row[col]
        else:
            lift[col] = np.nan
    lift = lift.drop('Random')
    return lift.sort_values('hit_rate', ascending=False)


def print_business_summary(results_df):
    """CEO-friendly summary of the business value."""
    best_model = results_df['hit_rate'].idxmax()
    best_hit = results_df.loc[best_model, 'hit_rate']
    random_hit = results_df.loc['Random', 'hit_rate']
    hit_lift = best_hit / random_hit if random_hit > 0 else float('inf')

    best_prec = results_df.loc[best_model, 'precision@10']
    random_prec = results_df.loc['Random', 'precision@10']
    prec_lift = best_prec / random_prec if random_prec > 0 else float('inf')

    print("=" * 60)
    print("EXECUTIVE SUMMARY — Recommender System Business Value")
    print("=" * 60)
    print(f"\nBest model: {best_model}")
    print(f"\nHit Rate (chance user gets at least 1 good rec):")
    print(f"  Random:       {random_hit:.1%}")
    print(f"  {best_model}: {best_hit:.1%}")
    print(f"  → {hit_lift:.1f}x improvement")
    print(f"\nPrecision@10 (fraction of recs that are relevant):")
    print(f"  Random:       {random_prec:.2%}")
    print(f"  {best_model}: {best_prec:.2%}")
    print(f"  → {prec_lift:.1f}x improvement")
    print(f"\nBottom line:")
    print(f"  Our system is ~{hit_lift:.0f}x more likely to show products")
    print(f"  users actually want vs the current random approach.")
    print("=" * 60)


def plot_model_comparison(results_df, out_dir='outputs/figures'):
    """Bar charts comparing all models across the three metrics."""
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ['precision@10', 'recall@10', 'hit_rate']
    titles = ['Precision@10', 'Recall@10', 'Hit Rate']
    colors = sns.color_palette('viridis', len(results_df))

    for ax, metric, title in zip(axes, metrics, titles):
        data = results_df[metric].sort_values(ascending=True)
        bars = ax.barh(data.index, data.values, color=colors)
        ax.set_xlabel(title)
        ax.set_title(title)
        for bar, val in zip(bars, data.values):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=8)

    plt.suptitle('Model Comparison — All Recommenders', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("saved model_comparison.png")


def plot_business_lift(lift_df, out_dir='outputs/figures'):
    """Bar chart of hit rate lift — the CEO-friendly chart."""
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    data = lift_df['hit_rate'].sort_values(ascending=True)
    colors = ['#2ecc71' if v == data.max() else '#3498db' for v in data.values]
    bars = ax.barh(data.index, data.values, color=colors)

    for bar, val in zip(bars, data.values):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}x', va='center', fontsize=11, fontweight='bold')

    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Random baseline (1x)')
    ax.set_xlabel('Lift over Random (times better)')
    ax.set_title('Hit Rate Improvement Over Random Recommendations', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'business_lift.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("saved business_lift.png")


def plot_weight_sensitivity(train, test, movies, user_item_matrix, item_similarity, cb_sim_df,
                            out_dir='outputs/figures'):
    """Sweep the CF weight in the weighted hybrid and plot the results."""
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(42)

    weights = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    rows = []

    for w in weights:
        print(f"  w_cf={w}...")
        def make_fn(w_val):
            def fn(uid, train, movies, n=10):
                return recommend_weighted_hybrid(uid, train, movies, user_item_matrix,
                                                 item_similarity, cb_sim_df, n=n, w_cf=w_val)
            return fn
        res = evaluate_hybrid(make_fn(w), train, test, movies, n=10, sample_size=100)
        res['w_cf'] = w
        rows.append(res)

    wdf = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(wdf['w_cf'], wdf['precision@10'], 'o-', label='Precision@10', linewidth=2)
    ax.plot(wdf['w_cf'], wdf['recall@10'], 's-', label='Recall@10', linewidth=2)
    ax.plot(wdf['w_cf'], wdf['hit_rate'], '^-', label='Hit Rate', linewidth=2)
    ax.set_xlabel('CF Weight (0 = pure content, 1 = pure CF)')
    ax.set_ylabel('Score')
    ax.set_title('Weighted Hybrid — Effect of CF Weight', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'weight_sensitivity.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("saved weight_sensitivity.png")
    return wdf


if __name__ == "__main__":
    ratings, movies, users = load_all()
    train, test = time_based_split(ratings, test_size=0.2)

    # build the models
    print("building user-item matrix...")
    user_item_matrix = build_user_item_matrix(train)

    print("computing item-item similarity...")
    item_similarity = compute_item_similarity(user_item_matrix)

    print("computing user-user similarity...")
    user_similarity = compute_user_similarity(user_item_matrix)

    print("building BoW content similarity...")
    corpus = build_corpus(movies)
    bow_matrix, vectorizer = build_bow_matrix(corpus)
    cb_sim_df = compute_bow_similarity(bow_matrix, movies)

    # full evaluation
    print("\n--- running full evaluation ---")
    results = run_full_evaluation(train, test, movies, users,
                                  user_item_matrix, item_similarity, user_similarity, cb_sim_df)

    results_df = build_comparison_table(results)
    print("\n" + results_df.to_string())

    # save comparison table
    os.makedirs('outputs', exist_ok=True)
    results_df.to_csv('outputs/model_comparison.csv')
    print("\nsaved model_comparison.csv")

    # business value
    lift_df = compute_lift(results_df)
    print("\nlift over random:")
    print(lift_df.round(2).to_string())
    print_business_summary(results_df)

    # plots
    print("\ngenerating plots...")
    plot_model_comparison(results_df)
    plot_business_lift(lift_df)

    print("\nweight sensitivity analysis...")
    plot_weight_sensitivity(train, test, movies, user_item_matrix, item_similarity, cb_sim_df)

    print("\ndone")
