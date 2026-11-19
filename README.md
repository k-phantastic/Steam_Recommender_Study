# Steam Recommender Study

Two predictive studies on 7.8 million Steam reviews from the UCSD Steam dataset:
1. **Playtime:** predict how many hours a reviewer played from their review.
2. **Genre tags:** predict a game's genres from review text alone.

**Team:** Khanh Phan, Danny Xia, Layth Marabeh<br>
**Maintained by:** Khanh Phan<br>
**Course:** DSC 256R, Data Mining on the Web, UC San Diego
_studies_results.png)

## Data
- [UCSD Steam datasets (McAuley Lab)](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data): **7.79M reviews** (`steam_reviews.json.gz`, downloaded separately because of its size), metadata for **32,135 games** (`steam_games.json.gz`), and bundle data (`bundle_data.json.gz`, used in EDA only)
- Steam's 22 genres were consolidated into **13 gameplay genres**. Software categories such as Accounting and Photo Editing were grouped into "Productivity/Software."

## Study 1: Playtime
**Question:** can review text and metadata predict a reviewer's playtime?

- **Target:** `log(1 + hours)`, to reduce the effect of extreme playtimes.
- **Features:** TF-IDF on cleaned review text (1,000 terms), plus scaled metadata such as review date and products owned.
- **Models:** an average baseline, Ridge, and XGBoost. The final model averages XGBoost and Ridge, then fits matrix factorization on the remaining error to capture user and game effects.

| Model (validation set, 60/20/20 split) | MSE on log hours | R² |
|:--|--:|--:|
| Predict the average | 3.18 | 0.00 |
| Ridge only | 2.62 | 0.18 |
| XGBoost only | 2.59 | 0.19 |
| **XGBoost + Ridge average** | **2.57** | **0.19** |

- **About 19% lower error than predicting the average,** with nearly the same result on the test set (2.568). Review text carries a real but limited signal about playtime.
- **Matrix factorization on the residuals added no improvement in our implementation.** The final score equals the XGBoost + Ridge average exactly. The user and game indices built for training didn't match the ones used at prediction time. Fixing that is the next step.
- An earlier baseline notebook (`10_...`) used a larger 7.79M-row set and an 80/10/10 split. There, Ridge reduced MSE from 3.13 to 2.61 compared with the average. Those numbers aren't directly comparable with the table above.

## Study 2: Genre tags
**Question:** can review text alone predict which of 13 genres a game belongs to?

- **Labels:** each review is tagged with its game's genres, encoded as 13 binary labels. The data is split 70/15/15.
- **Baseline:** keyword rules for each genre.
- **Model:** TF-IDF (5,000 terms) with a one-vs-rest LinearSVC. `C` was tuned by 3-fold grid search, first on a 100K-row sample, then on the full 5.5M-row training set.

| Model (test set) | Jaccard | Micro-F1 |
|:--|--:|--:|
| Keyword rules | 0.21 | 0.32 |
| TF-IDF + LinearSVC, 100K sample | 0.47 | 0.58 |
| **TF-IDF + LinearSVC, full data** | **0.49** | **0.59** |

- **The model reaches 2.3 times the keyword baseline's Jaccard score.**
- Predictions are precise (micro-precision 0.75) but miss many labels (recall 0.49). Rare genres such as Racing and Sports pull down the macro scores.

## Repository

| Notebook | Contents | Author |
|:--|:--|:--|
| `00_eda_games_reviews_bundles.ipynb` | EDA of game metadata, reviews and bundles | Khanh Phan |
| `01_eda_reviews_playtime.ipynb` | EDA of reviews and hours played | Layth Marabeh |
| `10_playtime_baseline_ridge.ipynb` | Study 1: average baseline vs. Ridge | Layth Marabeh |
| `11_playtime_xgboost_hybrid.ipynb` | Study 1: XGBoost and the hybrid model, with plots | Danny Xia |
| `20_genre_sample_100k.ipynb` | Study 2: development run on a 100K sample | Khanh Phan |
| `21_genre_full_model.ipynb` | Study 2: final model on the full data | Khanh Phan |

Notebook names follow `NN_study_step`: `0x` = shared EDA, `1x` = Study 1, `2x` = Study 2. Charts are in `images/`.

## Built with
`Python` `pandas` `NumPy` `scikit-learn` `XGBoost` `SciPy` `Matplotlib` `Seaborn` `Plotly`
