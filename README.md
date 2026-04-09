# Movie Recommender System

A full recommender system pipeline built on the MovieLens 1M dataset. We implement and evaluate multiple approaches, from simple baselines to hybrid models, and compare them to show the business value of personalized recommendations over random product display.

## Team

- Jack
- Omar
- Salmane
- Leena

## Dataset

[MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) — 1 million ratings from 6040 users on 3706 movies.

Download `ml-1m.zip`, unzip it, and place the three `.dat` files in `data/raw/` before running anything.

## Setup

```bash
git clone https://github.com/Jack-723/movie-recommender
cd movie-recommender
pip install -r requirements.txt
```

## Project Structure

```
movie-recommender/
├── data/
│   └── raw/                        # ratings.dat, movies.dat, users.dat (not committed)
├── outputs/
│   ├── figures/                    # all plots
│   ├── model_comparison.csv        # evaluation results for all models
│   ├── user_user_predictions.csv
│   └── item_item_predictions.csv
├── src/
│   ├── utils.py                    # data loading
│   ├── data_exploration.py         # exploration and analysis
│   ├── nonpersonalized.py          # random, popularity, demographic
│   ├── collaborative_filtering.py  # user-user, item-item, SVD
│   ├── content_based.py            # BoW, TF-IDF, lemmatization
│   ├── content_based_bert.py       # BERT embeddings
│   ├── hybrid_recommender.py       # weighted and switching hybrids
│   └── evaluation.py               # full evaluation pipeline across all models
├── .gitignore
├── requirements.txt
└── README.md
```

## How to Run

Each file can be run independently from the project root:

```bash
python src/data_exploration.py
python src/nonpersonalized.py
python src/collaborative_filtering.py
python src/content_based.py
python src/hybrid_recommender.py
python src/evaluation.py        # runs everything and produces all results
```

The main file to run is `evaluation.py`. It builds all models, evaluates them on the same test set, and saves the comparison table and plots to `outputs/`.

Note: `evaluation.py` takes a while to run, mainly due to SVD training and the hybrid evaluations. Expect around 30-60 minutes depending on your machine.

## Approaches Implemented

**Non-personalized**
- Random recommender (baseline)
- Popularity-based
- Demographic filtering (by gender, age, occupation)

**Collaborative Filtering**
- User-user memory-based CF
- Item-item memory-based CF
- SVD matrix factorization (model-based)

**Content-Based**
- Bag of Words with lemmatization
- TF-IDF
- BERT embeddings (sentence-transformers)

**Hybrid**
- Weighted hybrid (item-item CF + content-based)
- Switching hybrid (CF for warm users, content-based for cold start)

## Results

Best model: **SVD** with a hit rate of 30.5%, which is 15x better than random recommendations. Full results are in `outputs/model_comparison.csv`.

| Model | Precision@10 | Hit Rate | Lift over Random |
|-------|-------------|----------|-----------------|
| SVD | 0.0585 | 0.305 | 15.2x |
| Demographic (gender) | 0.0310 | 0.225 | 11.2x |
| Popularity | 0.0310 | 0.235 | 11.7x |
| Weighted Hybrid | 0.0260 | 0.155 | 7.8x |
| Item-Item CF | 0.0240 | 0.165 | 8.2x |
| Content-Based (BoW) | 0.0115 | 0.105 | 5.2x |
| Random | 0.0020 | 0.020 | 1x |
