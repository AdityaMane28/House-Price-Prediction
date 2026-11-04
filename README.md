# House Price Prediction: California Housing

An end-to-end machine learning pipeline that predicts **median house value** from the California Housing dataset. The project applies a **log transformation** to the skewed price target, **handles outliers** at both the target and feature level, and **benchmarks two gradient boosting models**: scikit-learn `GradientBoostingRegressor` (GBR) and `XGBoost`.

## Results

Metrics are computed on the hold-out test set (20% of the cleaned data) and reported in **original dollars**, not log units.

| Model   | CV RMSE (5-fold) | Test RMSE | Test MAE | Test R² | Test MAPE |
|---------|-----------------:|----------:|---------:|--------:|----------:|
| XGBoost | 39,972           | 39,243    | 25,847   | 0.818   | 14.5%     |
| GBR     | 42,787           | 42,113    | 28,423   | 0.790   | 15.9%     |

**XGBoost is the best model**, with roughly 7% lower RMSE than GBR on the test set. Cross-validation and test scores are close, so there is no sign of overfitting.

## Pipeline Overview

1. **Load and inspect:** 20,640 rows and 10 columns; `total_bedrooms` has 207 missing values.
2. **Log transform:** `median_house_value` skew drops from **0.978 to -0.173** after `log1p`. Models train on `log1p(price)`, and `TransformedTargetRegressor` converts predictions back with `expm1`.
3. **Outlier handling**
   - *Target:* removed 965 rows with the censored price cap (500,001) and trimmed the 1% / 99% tails of log-price, leaving 19,284 rows.
   - *Features:* an `IQRCapper` custom transformer winsorises values beyond 3 × IQR. Bounds are learned on **training data only**, so there is no leakage.
4. **Feature engineering:** `rooms_per_household`, `bedrooms_per_room`, `population_per_household`.
5. **Preprocessing:** median imputation, IQR capping, and one-hot encoding of `ocean_proximity`, all inside one `ColumnTransformer`.
6. **Modelling:** GBR and XGBoost, each wrapped in `Pipeline` and `TransformedTargetRegressor`.
7. **Evaluation:** 5-fold CV, then test-set RMSE, MAE, R² and MAPE, plus actual-vs-predicted plots, residual plots and feature importance.
8. **Persistence:** the best model is saved with `joblib` and accepts raw, unprocessed rows.

## Repository Structure

```
.
├── House_Price_Prediction.ipynb   # full pipeline, runs top to bottom
├── housing.csv                    # dataset (add this file, see below)
├── requirements.txt
└── README.md
```

## Dataset

The notebook expects a file named `housing.csv` in the project root, with these columns:

`longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, median_house_value, ocean_proximity`

This is the widely used **California Housing Prices** dataset (derived from the 1990 US Census). Download it, for example from Kaggle, and place it next to the notebook. Check the dataset's license before redistributing it in your own repository.

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook House_Price_Prediction.ipynb
```

Run the cells from top to bottom, because later cells depend on variables created earlier. The 5-fold CV cell is the slowest and can take a minute or two.

## Using the Saved Model

Running the final notebook cell creates `best_housing_model.joblib`. The saved object is the full pipeline, so it takes raw rows with the same columns the notebook trains on:

```python
import joblib
import pandas as pd

model = joblib.load("best_housing_model.joblib")

# X_new needs the 8 raw numeric columns, ocean_proximity, and the three engineered
# features (rooms_per_household, bedrooms_per_room, population_per_household)
predictions = model.predict(X_new)   # predicted prices in dollars
```

Note that the engineered features are created in the notebook before the train/test split, so new data needs them added in the same way before calling `predict`.

## Key Findings

- `ocean_proximity` (especially `INLAND`) and `median_income` are the most influential features for the XGBoost model.
- Prediction errors are larger for very cheap and very expensive districts, as shown in the residual plots.

## Limitations

- Rows at the dataset's price cap (500,001) were removed, so the model is **not reliable for homes above roughly $465k**.
- The data is from the 1990 census and describes block groups, not individual houses, so predictions are not suitable for real-world valuation.
- Hyper-parameters are set by hand and not tuned. `RandomizedSearchCV` would likely improve the scores.

## Possible Improvements

- Hyper-parameter tuning (`RandomizedSearchCV` or Optuna)
- Add LightGBM / CatBoost and a Random Forest or Ridge baseline
- SHAP values for model explainability
- Wrap the model in a small Streamlit or FastAPI app for predictions

## Tech Stack

Python · pandas · NumPy · SciPy · scikit-learn · XGBoost · Matplotlib · Seaborn · joblib · Jupyter
