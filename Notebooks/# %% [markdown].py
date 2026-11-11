# %% [markdown]
# # House Prices — Modeling Notebook
# Predict house prices and evaluate models with RMSLE. This notebook separates numeric polynomial features from one-hot encoded categorical features.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor

# %% [markdown]
# ## 1. Load data

# %%
data = pd.read_csv('../Data/raw/train.csv')
training_data = data.copy()
display(training_data.head())
training_data.info()

# %% [markdown]
# ## 2. Exploratory data analysis

# %%
eda_data = training_data.drop(columns=['Id'], errors='ignore')
display(eda_data.isnull().sum().sort_values(ascending=False).head(20))
display(eda_data.select_dtypes(include=np.number).skew().sort_values(ascending=False).head(20))
display(eda_data.corr(numeric_only=True)['SalePrice'].sort_values(ascending=False))

# %%
eda_data.hist(bins=40, figsize=(20, 15))
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Define cleaning and ordinal mappings

# %%
none_cols = [
    'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
    'BsmtFinType2', 'MasVnrType'
]
zero_cols = ['GarageYrBlt', 'MasVnrArea']

qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
bsmt_exposure_map = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0}
bsmt_fin_map = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0}
garage_finish_map = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'None': 0}
paved_drive_map = {'Y': 2, 'P': 1, 'N': 0}
functional_map = {'Typ': 7, 'Min1': 6, 'Min2': 5, 'Mod': 4, 'Maj1': 3, 'Maj2': 2, 'Sev': 1, 'Sal': 0}
fence_map = {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'None': 0}
utilities_map = {'AllPub': 3, 'NoSewr': 2, 'NoSeWa': 1, 'ELO': 0}

ordinal_mappings = {
    **{col: qual_map for col in [
        'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
        'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC'
    ]},
    'BsmtExposure': bsmt_exposure_map,
    'BsmtFinType1': bsmt_fin_map,
    'BsmtFinType2': bsmt_fin_map,
    'GarageFinish': garage_finish_map,
    'PavedDrive': paved_drive_map,
    'Functional': functional_map,
    'Fence': fence_map,
    'Utilities': utilities_map,
}

# %% [markdown]
# ## 4. Custom transformers

# %%
class HousePriceCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, none_cols, zero_cols, ordinal_mappings):
        self.none_cols = none_cols
        self.zero_cols = zero_cols
        self.ordinal_mappings = ordinal_mappings

    def fit(self, X, y=None):
        X = X.copy()
        self.neighborhood_medians_ = X.groupby('Neighborhood')['LotFrontage'].median()
        self.global_median_ = X['LotFrontage'].median()
        self.electrical_mode_ = X['Electrical'].mode().iloc[0]
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.none_cols:
            if col in X:
                X[col] = X[col].fillna('None')

        X['HasGarage'] = (X['GarageType'] != 'None').astype(int)

        for col in self.zero_cols:
            if col in X:
                X[col] = X[col].fillna(0)

        neighborhood_medians = X['Neighborhood'].map(self.neighborhood_medians_)
        X['LotFrontage'] = X['LotFrontage'].fillna(neighborhood_medians)
        X['LotFrontage'] = X['LotFrontage'].fillna(self.global_median_)
        X['Electrical'] = X['Electrical'].fillna(self.electrical_mode_)

        for col in ['MSSubClass', 'MoSold']:
            X[col] = X[col].astype(str)

        for col, mapping in self.ordinal_mappings.items():
            if col not in X:
                continue
            encoded = X[col].map(mapping)
            if encoded.isna().any():
                unknown = X.loc[encoded.isna(), col].unique()
                raise ValueError(f'Unknown categories in {col}: {unknown}')
            X[col] = encoded
        return X


class HousePriceFeatureEngineer(BaseEstimator, TransformerMixin):

    ALL_FEATURES = [
        # Area
        'TotalSF',
        'TotalPorchSF',
        'TotalOutdoorSF',

        # Bathrooms
        'TotalBathrooms',
        'TotalFullBath',
        'TotalHalfBath',

        # Age
        'HouseAge',
        'RemodAge',
        'GarageAge',
        'YearsSinceRemod',

        # Quality
        'OverallScore',
        'ExteriorScore',
        'GarageScore',
        'BasementScore',
        'TotalQuality',

        # Garage
        'GarageAreaPerCar',
        'HasGarage',

        # Basement
        'TotalBsmtFinishedSF',
        'BsmtFinishedRatio',
        'BsmtUnfinishedRatio',
        'HasBasement',

        # Lot
        'LotUtilization',
        'LotFrontageRatio',

        # Rooms
        'RoomsPerBedroom',
        'AreaPerRoom',
        'AreaPerBedroom',

        # Binary
        'HasPool',
        'HasFireplace',
        'Has2ndFloor',
        'HasWoodDeck',
        'HasMasonryVeneer',
        'HasCentralAir',
    ]

    def __init__(self, features=None):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        selected_features = (
            self.ALL_FEATURES
            if self.features is None
            else self.features
        )

        # Helper function to safely calculate ratios
        def safe_divide(numerator, denominator):
            return np.divide(
                numerator,
                denominator,
                out=np.zeros(len(X), dtype=float),
                where=np.asarray(denominator) != 0
            )

        # 1. Area Features

        if 'TotalSF' in selected_features:
            X['TotalSF'] = (
                X['TotalBsmtSF'] +
                X['1stFlrSF'] +
                X['2ndFlrSF']
            )

        if 'TotalPorchSF' in selected_features:
            X['TotalPorchSF'] = (
                X['OpenPorchSF'] +
                X['EnclosedPorch'] +
                X['3SsnPorch'] +
                X['ScreenPorch']
            )

        if 'TotalOutdoorSF' in selected_features:
            X['TotalOutdoorSF'] = (
                X['WoodDeckSF'] +
                X['OpenPorchSF'] +
                X['EnclosedPorch'] +
                X['3SsnPorch'] +
                X['ScreenPorch']
            )

        # 2. Bathroom Features

        if 'TotalBathrooms' in selected_features:
            X['TotalBathrooms'] = (
                X['FullBath'] +
                0.5 * X['HalfBath'] +
                X['BsmtFullBath'] +
                0.5 * X['BsmtHalfBath']
            )

        if 'TotalFullBath' in selected_features:
            X['TotalFullBath'] = (
                X['FullBath'] +
                X['BsmtFullBath']
            )

        if 'TotalHalfBath' in selected_features:
            X['TotalHalfBath'] = (
                X['HalfBath'] +
                X['BsmtHalfBath']
            )

        # 3. Age Features

        if 'HouseAge' in selected_features:
            X['HouseAge'] = X['YrSold'] - X['YearBuilt']

        if 'RemodAge' in selected_features:
            X['RemodAge'] = X['YrSold'] - X['YearRemodAdd']

        if 'GarageAge' in selected_features:
            X['GarageAge'] = np.where(
                X['GarageYrBlt'] > 0,
                X['YrSold'] - X['GarageYrBlt'],
                0
            )

        if 'YearsSinceRemod' in selected_features:
            X['YearsSinceRemod'] = (
                X['YearRemodAdd'] - X['YearBuilt']
            )

        # 4. Quality Features
        # Ordinal features are converted to numeric by HousePriceCleaner.

        if 'OverallScore' in selected_features:
            X['OverallScore'] = (
                X['OverallQual'] * X['OverallCond']
            )

        if 'ExteriorScore' in selected_features:
            X['ExteriorScore'] = (
                X['ExterQual'] * X['ExterCond']
            )

        if 'GarageScore' in selected_features:
            X['GarageScore'] = (
                X['GarageQual'] * X['GarageCond']
            )

        if 'BasementScore' in selected_features:
            X['BasementScore'] = (
                X['BsmtQual'] * X['BsmtCond']
            )

        if 'TotalQuality' in selected_features:
            X['TotalQuality'] = (
                X['OverallQual'] +
                X['ExterQual'] +
                X['KitchenQual']
            )

        # 5. Garage Features

        if 'GarageAreaPerCar' in selected_features:
            X['GarageAreaPerCar'] = safe_divide(
                X['GarageArea'],
                X['GarageCars']
            )

        if 'HasGarage' in selected_features:
            X['HasGarage'] = (
                X['GarageType'] != 'None'
            ).astype(int)

        # 6. Basement Features

        if 'TotalBsmtFinishedSF' in selected_features:
            X['TotalBsmtFinishedSF'] = (
                X['BsmtFinSF1'] +
                X['BsmtFinSF2']
            )

        if 'BsmtFinishedRatio' in selected_features:
            X['BsmtFinishedRatio'] = safe_divide(
                X['BsmtFinSF1'] + X['BsmtFinSF2'],
                X['TotalBsmtSF']
            )

        if 'BsmtUnfinishedRatio' in selected_features:
            X['BsmtUnfinishedRatio'] = safe_divide(
                X['BsmtUnfSF'],
                X['TotalBsmtSF']
            )

        if 'HasBasement' in selected_features:
            X['HasBasement'] = (
                X['TotalBsmtSF'] > 0
            ).astype(int)

        # 7. Lot Features

        if 'LotUtilization' in selected_features:
            X['LotUtilization'] = safe_divide(
                X['GrLivArea'],
                X['LotArea']
            )

        if 'LotFrontageRatio' in selected_features:
            X['LotFrontageRatio'] = safe_divide(
                X['LotFrontage'],
                X['LotArea']
            )

        # 8. Room Features

        if 'RoomsPerBedroom' in selected_features:
            X['RoomsPerBedroom'] = safe_divide(
                X['TotRmsAbvGrd'],
                X['BedroomAbvGr']
            )

        if 'AreaPerRoom' in selected_features:
            X['AreaPerRoom'] = safe_divide(
                X['GrLivArea'],
                X['TotRmsAbvGrd']
            )

        if 'AreaPerBedroom' in selected_features:
            X['AreaPerBedroom'] = safe_divide(
                X['GrLivArea'],
                X['BedroomAbvGr']
            )

        # 9. Binary Features

        binary_features = {
            'HasPool': X['PoolArea'] > 0,
            'HasFireplace': X['Fireplaces'] > 0,
            'Has2ndFloor': X['2ndFlrSF'] > 0,
            'HasWoodDeck': X['WoodDeckSF'] > 0,
            'HasMasonryVeneer': X['MasVnrArea'] > 0,
            'HasCentralAir': X['CentralAir'] == 'Y',
        }

        for feature, condition in binary_features.items():
            if feature in selected_features:
                X[feature] = condition.astype(int)

        # Validate feature names
        unknown_features = (
            set(selected_features) - set(self.ALL_FEATURES)
        )

        if unknown_features:
            raise ValueError(
                f'Unknown features: {sorted(unknown_features)}'
            )

        return X


class HousePriceSkewnessTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, skew_threshold=1.0, exclude=None):
        self.skew_threshold = skew_threshold
        self.exclude = exclude

    def fit(self, X, y=None):
        numeric_cols = X.select_dtypes(include=np.number).columns
        excluded = [] if self.exclude is None else list(self.exclude)
        self.log_features_ = []
        for col in numeric_cols:
            if col in excluded or X[col].nunique(dropna=True) <= 2:
                continue
            if X[col].min(skipna=True) < 0:
                continue
            if X[col].skew() > self.skew_threshold:
                self.log_features_.append(col)
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.log_features_:
            # Guard against negative values appearing in later data.
            if (X[col] < 0).any():
                raise ValueError(f'{col} contains negative values; log1p is not valid.')
            X[col] = np.log1p(X[col])
        return X

# %% [markdown]
# ## 5. Build preprocessing
# **Important fix:** `PolynomialFeatures` is placed only in the numeric branch of the `ColumnTransformer`. The categorical branch applies `OneHotEncoder` directly, so dummy variables never get polynomial-expanded. Numeric/categorical selectors run after cleaning and ordinal mapping.

# %%
# Baseline preprocessing: feature engineering is intentionally disabled here.
# evaluate_model() adds HousePriceFeatureEngineer only when requested.

numeric_pipeline = Pipeline([
    ('polynomial', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
])

categorical_pipeline = OneHotEncoder(
    handle_unknown='ignore',
    sparse_output=False
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, make_column_selector(dtype_include=np.number)),
        ('cat', categorical_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder='drop',
    verbose_feature_names_out=True,
)

preprocessing = Pipeline([
    ('cleaner', HousePriceCleaner(none_cols, zero_cols, ordinal_mappings)),
    ('skewness', HousePriceSkewnessTransformer(skew_threshold=1.0)),
    ('preprocessor', preprocessor),
])

# %% [markdown]
# ## 6. Reusable model evaluation

# %%
def evaluate_model(
    model,
    X,
    y,
    cv=5,
    polynomial=True,
    degree=2,
    scaler=True,
    feature_engineering=False,
    engineered_features=None,
    verbose=True,
    return_estimator=False
):
    """Cross-validate a regressor using RMSLE.

    feature_engineering=False:
        Use the original cleaned features only.

    feature_engineering=True:
        Use HousePriceFeatureEngineer.DEFAULT_FEATURES.

    engineered_features=[...]:
        Use only the listed engineered features. This option automatically
        enables feature engineering.

    polynomial and scaler can be controlled independently.
    """

    # Numeric preprocessing
    numeric_steps = [
        ('polynomial', PolynomialFeatures(
            degree=degree if polynomial else 1,
            include_bias=False
        ))
    ]

    if scaler:
        numeric_steps.append(('scaler', StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)

    custom_preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, make_column_selector(dtype_include=np.number)),
            ('cat', OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False
            ), make_column_selector(dtype_include=object)),
        ],
        remainder='drop',
        verbose_feature_names_out=True,
    )

    # Build preprocessing steps in the correct order.
    prep_steps = [
        ('cleaner', HousePriceCleaner(
            none_cols, zero_cols, ordinal_mappings
        ))
    ]

    use_feature_engineering = (
        feature_engineering or engineered_features is not None
    )

    if use_feature_engineering:
        prep_steps.append((
            'feature_engineering',
            HousePriceFeatureEngineer(features=engineered_features)
        ))

    prep_steps.extend([
        ('skewness', HousePriceSkewnessTransformer(skew_threshold=1.0)),
        ('preprocessor', custom_preprocessor),
    ])

    prep = Pipeline(prep_steps)

    model_pipeline = Pipeline([
        ('preprocessing', prep),
        ('model', model),
    ])

    final_model = TransformedTargetRegressor(
        regressor=model_pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )

    if return_estimator:
        return final_model

    scores = -cross_val_score(
        final_model, X, y,
        cv=cv,
        scoring='neg_root_mean_squared_log_error',
        n_jobs=-1,
    )

    if verbose:
        print('RMSLE scores:', scores)
        print(f'Mean RMSLE: {scores.mean():.6f}')
        print(f'Std RMSLE:  {scores.std():.6f}')

    return scores

# %% [markdown]
# ## 7. Prepare features and target

# %%
X = data.drop(columns=['SalePrice', 'Id'])
y = data['SalePrice']

# %% [markdown]
# ## 8. Evaluate models

# %%
linear_scores = evaluate_model(LinearRegression(), X, y, polynomial=False)

# %%
sgd_model = SGDRegressor(
    max_iter=10000, tol=1e-5, penalty=None,
    eta0=0.01, n_iter_no_change=100, random_state=42
)
sgd_scores = evaluate_model(sgd_model, X, y, polynomial=False)

# %% [markdown]
# ### Ridge: compare alpha and polynomial degree

# %%
ridge_results = []
for degree in [1, 2]:
    for alpha in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        scores = evaluate_model(
            Ridge(alpha=alpha), X, y,
            polynomial=(degree > 1), degree=degree, verbose=False
        )
        ridge_results.append({
            'degree': degree, 'alpha': alpha,
            'mean_rmsle': scores.mean(), 'std_rmsle': scores.std()
        })

ridge_results_df = pd.DataFrame(ridge_results).sort_values('mean_rmsle')
display(ridge_results_df)

# %% [markdown]
# ### Lasso: compare alpha

# %%
'''lasso_results = []
for polynomial, degree, alphas in [
    (False, 1, [0.0001, 0.001, 0.01, 0.1, 1]),
    (True, 2, [0.0001, 0.0003, 0.0005, 0.0007, 0.001, 0.0015, 0.002]),
]:
    for alpha in alphas:
        print(f'Testing alpha={alpha}, polynomial={polynomial}')
        model = Lasso(alpha=alpha, max_iter=20000, tol=1e-3)
        scores = evaluate_model(
            model, X, y, polynomial=polynomial, degree=degree, verbose=False
        )
        lasso_results.append({
            'polynomial': polynomial, 'degree': degree, 'alpha': alpha,
            'mean_rmsle': scores.mean(), 'std_rmsle': scores.std()
        })

lasso_results_df = pd.DataFrame(lasso_results).sort_values('mean_rmsle')
display(lasso_results_df)'''

# %% [markdown]
# ## Notes
# - RMSLE is computed on the original target scale; `TransformedTargetRegressor` applies `log1p` during fitting and `expm1` for inverse transformation.
# - All learned preprocessing is inside the cross-validation pipeline to avoid leakage.
# - The numeric polynomial branch can expand many numeric features and may be memory-intensive. Start with degree 2 and compare against `polynomial=False`.
# - `nominal_cols` was removed because the column selector already identifies object columns after cleaning. If you want a fixed categorical feature list, pass that list to `ColumnTransformer` instead.
# - Spelling fixes include *columns*, *custom*, *transformation*, *gradient*, and *polynomial*.

# %% [markdown]
# ### Elastic Net Regression: compare alpha

# %%
from sklearn.linear_model import ElasticNet
import pandas as pd

elastic_net_results = []

alphas = [0.0005, 0.001, 0.003, 0.01, 0.03]
l1_ratios = [0.2, 0.5, 0.8]

for l1_ratio in l1_ratios:
    for alpha in alphas:

        print(f"Testing alpha={alpha}, l1_ratio={l1_ratio}")

        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=20000,
            tol=1e-3,
            random_state=42
        )

        scores = evaluate_model(
            model,
            X,
            y,
            polynomial=False,
            cv=3,
            verbose=False
        )

        elastic_net_results.append({
            'alpha': alpha,
            'l1_ratio': l1_ratio,
            'mean_rmsle': scores.mean(),
            'std_rmsle': scores.std()
        })

elastic_net_results_df = pd.DataFrame(elastic_net_results)

elastic_net_results_df = elastic_net_results_df.sort_values(
    'mean_rmsle'
).reset_index(drop=True)

display(elastic_net_results_df)

print("\nBest Model:")
display(elastic_net_results_df.head(1))

# %%
elastic_net_fine_results = []

alphas = [0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.001]
l1_ratios = [0.7, 0.8, 0.9, 0.95]

for l1_ratio in l1_ratios:
    for alpha in alphas:

        print(f"Testing alpha={alpha}, l1_ratio={l1_ratio}")

        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=20000,
            tol=1e-3,
            random_state=42
        )

        scores = evaluate_model(
            model,
            X,
            y,
            polynomial=False,
            cv=5,
            verbose=False
        )

        elastic_net_fine_results.append({
            'alpha': alpha,
            'l1_ratio': l1_ratio,
            'mean_rmsle': scores.mean(),
            'std_rmsle': scores.std()
        })

elastic_net_fine_results_df = pd.DataFrame(
    elastic_net_fine_results
).sort_values('mean_rmsle').reset_index(drop=True)

display(elastic_net_fine_results_df)

# %%
best_elastic_net = ElasticNet(
    alpha=0.0006,
    l1_ratio=0.8,
    max_iter=20000,
    tol=1e-3,
    random_state=42
)

polynomial_scores = evaluate_model(
    best_elastic_net,
    X,
    y,
    polynomial=True,
    degree=2,
    cv=5,
    verbose=True
)

# %%
best_elastic_net = ElasticNet(
    alpha=0.0006,
    l1_ratio=0.8,
    max_iter=20000,
    tol=1e-3,
    random_state=42
)

elastic_net_scores = evaluate_model(
    best_elastic_net,
    X,
    y,
    polynomial=False,
    cv=5,
    verbose=True
)

# %% [markdown]
# ## 9. Feature Engineering (Planning)

# %% [markdown]
# ### Feature Engineering Ideas
# 
# ### 1. Area Features
# Combine area-related columns to capture the property's total usable space.
# 
# | New Feature | Formula |
# |---|---|
# | `TotalSF` | `TotalBsmtSF + 1stFlrSF + 2ndFlrSF` |
# | `TotalPorchSF` | `OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch` |
# | `TotalOutdoorSF` | `WoodDeckSF + OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch` |
# 
# ### 2. Bathroom Features
# Combine bathroom counts, accounting for half bathrooms.
# 
# | New Feature | Formula |
# |---|---|
# | `TotalBathrooms` | `FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath` |
# | `TotalFullBath` | `FullBath + BsmtFullBath` |
# | `TotalHalfBath` | `HalfBath + BsmtHalfBath` |
# 
# ### 3. Age Features
# Calculate the age of the house and its major components at the time of sale.
# 
# | New Feature | Formula |
# |---|---|
# | `HouseAge` | `YrSold - YearBuilt` |
# | `RemodAge` | `YrSold - YearRemodAdd` |
# | `GarageAge` | `YrSold - GarageYrBlt` |
# | `YearsSinceRemod` | `YearRemodAdd - YearBuilt` |
# 
# ### 4. Quality Features
# Combine ordinal quality ratings to represent overall property quality.
# 
# | New Feature | Formula |
# |---|---|
# | `OverallScore` | `OverallQual * OverallCond` |
# | `ExteriorScore` | `ExterQual * ExterCond` |
# | `GarageScore` | `GarageQual * GarageCond` |
# | `BasementScore` | `BsmtQual * BsmtCond` |
# | `TotalQuality` | `OverallQual + ExterQual + KitchenQual` |
# 
# *Ordinal categories must be converted to numerical values before applying these formulas.*
# 
# ### 5. Garage Features
# Create features describing garage capacity and availability.
# 
# | New Feature | Formula |
# |---|---|
# | `GarageAreaPerCar` | `GarageArea / GarageCars` |
# | `HasGarage` | `GarageType != 'None'` |
# 
# *Handle zero values in `GarageCars` to avoid division by zero. `HasGarage` is already created by the cleaner.*
# 
# ### 6. Basement Features
# Capture finished and unfinished basement areas.
# 
# | New Feature | Formula |
# |---|---|
# | `TotalBsmtFinishedSF` | `BsmtFinSF1 + BsmtFinSF2` |
# | `BsmtFinishedRatio` | `(BsmtFinSF1 + BsmtFinSF2) / TotalBsmtSF` |
# | `BsmtUnfinishedRatio` | `BsmtUnfSF / TotalBsmtSF` |
# | `HasBasement` | `TotalBsmtSF > 0` |
# 
# *Handle zero values in `TotalBsmtSF` when calculating ratios.*
# 
# ### 7. Lot Features
# Measure how efficiently the lot area is used.
# 
# | New Feature | Formula |
# |---|---|
# | `LotUtilization` | `GrLivArea / LotArea` |
# | `LotFrontageRatio` | `LotFrontage / LotArea` |
# 
# ### 8. Room Features
# Create ratios to describe room and living-area distribution.
# 
# | New Feature | Formula |
# |---|---|
# | `RoomsPerBedroom` | `TotRmsAbvGrd / Bedroom` |
# | `AreaPerRoom` | `GrLivArea / TotRmsAbvGrd` |
# | `AreaPerBedroom` | `GrLivArea / Bedroom` |
# 
# ### 9. Binary Features
# Convert the presence of certain property features into binary indicators.
# 
# | New Feature | Formula |
# |---|---|
# | `HasPool` | `PoolArea > 0` |
# | `HasFireplace` | `Fireplaces > 0` |
# | `Has2ndFloor` | `2ndFlrSF > 0` |
# | `HasWoodDeck` | `WoodDeckSF > 0` |
# | `HasMasonryVeneer` | `MasVnrArea > 0` |
# | `HasCentralAir` | `CentralAir == 'Y'` |
# 
# ---
# 
# ## Initial Features to Implement
# 
# Start with a smaller set of potentially useful features:
# 
# - `TotalSF`
# - `TotalPorchSF`
# - `TotalBathrooms`
# - `HouseAge`
# - `RemodAge`
# - `GarageAge`
# - `OverallScore`
# - `TotalQuality`
# - `TotalBsmtFinishedSF`
# - `BsmtFinishedRatio`
# - `LotUtilization`
# - `HasFireplace`
# 
# Evaluate their impact using cross-validation and RMSLE before adding more features.

# %% [markdown]
# ## 10. Decision Tree, Random Forest and Ensemble Models

# %% [markdown]
# ### Feature Engineering controls
# 
# Feature engineering is **off by default** in `evaluate_model()`, so Sections 1–9 use the original features.
# 
# Starting in Section 10, enable it explicitly:
# 
# ```python
# feature_engineering=True
# ```
# 
# To test only selected engineered features, pass `engineered_features`:
# 
# ```python
# engineered_features=['TotalSF', 'HouseAge', 'TotalQuality']
# ```
# 
# To disable it again, use `feature_engineering=False` and omit `engineered_features`.
# 
# The available feature names are documented in `HousePriceFeatureEngineer.DEFAULT_FEATURES`
# and the `HousePriceFeatureEngineer` class. Add, remove, or edit feature formulas there.
# 

# %%
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(
    random_state=42
)

tree_scores = evaluate_model(
    model=tree_reg,
    X=X,
    y=y,
    cv=5,
    polynomial=False,
    scaler=False,
    feature_engineering=True,
    verbose=True
)

# %%
tree_results = []

max_depths = [3, 5, 7, 10, 15, 20, None]
min_samples_leafs = [1, 2, 4, 8, 12]
min_samples_splits = [2, 5, 10, 20]

for depth in max_depths:
    for leaf in min_samples_leafs:
        for split in min_samples_splits:

            print(
                f"Testing depth={depth}, "
                f"leaf={leaf}, split={split}"
            )

            model = DecisionTreeRegressor(
                max_depth=depth,
                min_samples_leaf=leaf,
                min_samples_split=split,
                random_state=42
            )

            scores = evaluate_model(
                model=model,
                X=X,
                y=y,
                cv=5,
                polynomial=False,
                scaler=False,
                feature_engineering=True,
                verbose=False
            )

            tree_results.append({
                'max_depth': depth,
                'min_samples_leaf': leaf,
                'min_samples_split': split,
                'mean_rmsle': scores.mean(),
                'std_rmsle': scores.std()
            })

tree_results_df = pd.DataFrame(tree_results)

tree_results_df = tree_results_df.sort_values(
    'mean_rmsle'
).reset_index(drop=True)

display(tree_results_df.head(15))

# %%
best_params = tree_results_df.iloc[0]

best_depth = best_params['max_depth']

# Convert max_depth to int, or None
best_depth = (
    None if pd.isna(best_depth)
    else int(best_depth)
)

best_tree = DecisionTreeRegressor(
    max_depth=best_depth,
    min_samples_leaf=int(best_params['min_samples_leaf']),
    min_samples_split=int(best_params['min_samples_split']),
    random_state=42
)

best_tree_scores = evaluate_model(
    model=best_tree,
    X=X,
    y=y,
    cv=5,
    polynomial=False,
    scaler=False,
    feature_engineering=True,
    verbose=True
)

# %%
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_features=1.0,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

rf_scores = evaluate_model(
    model=rf_model,
    X=X,
    y=y,
    cv=5,
    polynomial=False,
    scaler=False,
    feature_engineering=True,
    verbose=True
)

# %%


# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# 1. Define Random Forest
rf = RandomForestRegressor(
    random_state=42,
    n_jobs=1
)

# 2. Get the existing pipeline from evaluate_model
rf_pipeline = evaluate_model(
    model=rf,
    X=X,
    y=y,
    polynomial=False,
    scaler=False,
    feature_engineering=True,
    verbose=False,
    return_estimator=True
)

# 3. Define hyperparameter distributions
param_distributions = {
    'regressor__model__n_estimators': randint(200, 1200),
    'regressor__model__max_features': [0.3, 0.5, 0.7, 0.9, 1.0],
    'regressor__model__max_depth': [None, 10, 15, 20, 25, 30],
    'regressor__model__min_samples_split': randint(2, 20),
    'regressor__model__min_samples_leaf': randint(1, 10),
    'regressor__model__max_samples': [0.6, 0.7, 0.8, 0.9, 1.0]
}

# 4. Randomized Search
random_search = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=param_distributions,
    n_iter=30,
    scoring='neg_root_mean_squared_log_error',
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=2,
    refit=True
)

# 5. Fit
random_search.fit(X, y)

# 6. Results
print(f"Best RMSLE: {-random_search.best_score_:.6f}")

print("\nBest Parameters:")
print(random_search.best_params_)

# 7. Best model
best_rf = random_search.best_estimator_

# %%
# Extract the best complete pipeline
best_model = random_search.best_estimator_

# Extract the fitted Random Forest estimator
best_rf = best_model.regressor_.named_steps['model']

# Display the best model
print("Best Random Forest Model:")
print(best_rf)

print("\nBest RMSLE:")
print(-random_search.best_score_)

print("\nBest Parameters:")
print(random_search.best_params_)

# %%
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# 1. Define Gradient Boosting
gbr = GradientBoostingRegressor(
    random_state=42
)

# 2. Get the existing preprocessing pipeline
gbr_pipeline = evaluate_model(
    model=gbr,
    X=X,
    y=y,
    polynomial=False,
    scaler=False,
    feature_engineering=True,
    verbose=False,
    return_estimator=True
)

# 3. Hyperparameter distributions
param_distributions = {
    'regressor__model__n_estimators': randint(100, 1000),

    'regressor__model__learning_rate': uniform(0.01, 0.19),

    'regressor__model__max_depth': randint(2, 6),

    'regressor__model__min_samples_split': randint(2, 20),

    'regressor__model__min_samples_leaf': randint(1, 10),

    'regressor__model__subsample': uniform(0.6, 0.4),

    'regressor__model__max_features': [None, 'sqrt', 'log2']
}

# 4. RandomizedSearchCV
gbr_search = RandomizedSearchCV(
    estimator=gbr_pipeline,
    param_distributions=param_distributions,
    n_iter=50,
    scoring='neg_root_mean_squared_log_error',
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=2,
    refit=True
)

# 5. Fit
gbr_search.fit(X, y)

# 6. Results
print(f"Best RMSLE: {-gbr_search.best_score_:.6f}")

print("\nBest Parameters:")
print(gbr_search.best_params_)

# 7. Extract best model
best_gbr_model = gbr_search.best_estimator_

best_gbr = best_gbr_model.regressor_.named_steps['model']

print("\nBest Gradient Boosting Model:")
print(best_gbr)

# %%
best_rmsle = -gbr_search.best_score_

print(f"Best CV RMSLE: {best_rmsle:.6f}")

# %%



