import { TrainingConfig, Hyperparameters } from './types.ts';

export function generatePythonScikitCode(config: TrainingConfig, hyperparams: Hyperparameters): string {
  return `"""
House Price Prediction – Machine Learning Regression Pipeline
Technologies: Python, Pandas, NumPy, Matplotlib, Scikit-learn

• End-to-end machine learning model to predict house prices from property-related features
• Data cleaning, exploratory data analysis, and feature engineering
• Model training: Linear Regression, Decision Tree Regressor, Random Forest Regressor
• Performance comparison using MAE, RMSE, and R² evaluation metrics
• Feature importance and residual analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. LOAD & EXPLORE DATASET
# ==========================================
# In production, load your CSV: df = pd.read_csv('housing_data.csv')
print("--- [1] Loading and Inspecting Housing Dataset ---")
df = pd.read_csv('housing_data.csv')
print(f"Dataset shape: {df.shape}")
print(df.info())
print(df.describe())

# ==========================================
# 2. DATA CLEANING & OUTLIER HANDLING
# ==========================================
print("\\n--- [2] Performing Data Cleaning & Outlier Removal ---")
${config.filterOutliers ? `
# Outlier Removal using Interquartile Range (IQR) on SalePrice
Q1 = df['salePrice'].quantile(0.25)
Q3 = df['salePrice'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - ${config.outlierThresholdIQR} * IQR
upper_bound = Q3 + ${config.outlierThresholdIQR} * IQR

initial_count = len(df)
df = df[(df['salePrice'] >= lower_bound) & (df['salePrice'] <= upper_bound)]
print(f"Removed {initial_count - len(df)} outliers outside [\${lower_bound:,.0f}, \${upper_bound:,.0f}]")
` : '# Outlier filtering was disabled in current configuration'}

# Check for missing values
missing_counts = df.isnull().sum()
if missing_counts.sum() > 0:
    print("Imputing missing values with median/mode:")
    df = df.fillna(df.median(numeric_only=True))

# ==========================================
# 3. FEATURE PREPARATION & ENCODING
# ==========================================
print("\\n--- [3] Feature Engineering & One-Hot Encoding ---")

# Categorical one-hot encoding for Neighborhood
df_encoded = pd.get_dummies(df, columns=['neighborhood'], drop_first=True)

# Separate Target and Features
X = df_encoded.drop(columns=['id', 'salePrice'], errors='ignore')
y = df_encoded['salePrice']

${config.logTransformTarget ? `
# Log-transform target to reduce heteroscedasticity and target skewness
y = np.log1p(y)
` : ''}

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=${(1 - config.trainRatio).toFixed(2)}, random_state=${config.randomSeed}
)
print(f"Training set: {X_train.shape[0]} samples, Testing set: {X_test.shape[0]} samples")

# Feature Standardization
${config.standardizeFeatures ? `
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
` : `
X_train_scaled = X_train
X_test_scaled = X_test
`}

# ==========================================
# 4. MODEL TRAINING & REGRESSION ALGORITHMS
# ==========================================
print("\\n--- [4] Training Regression Models ---")

models = {
    "Linear Regression (Ridge)": Ridge(alpha=${hyperparams.linearRidgeLambda}),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=${hyperparams.treeMaxDepth},
        min_samples_split=${hyperparams.treeMinSamplesSplit},
        random_state=${config.randomSeed}
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=${hyperparams.forestNumTrees},
        max_depth=${hyperparams.forestMaxDepth},
        max_features=${hyperparams.forestFeatureSubsampleRatio},
        random_state=${config.randomSeed},
        n_jobs=-1
    )
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    ${config.logTransformTarget ? `
    # Invert log transform for human-readable dollar evaluation
    y_test_eval = np.expm1(y_test)
    y_pred_eval = np.expm1(y_pred_test)
    y_train_eval = np.expm1(y_train)
    y_train_pred_eval = np.expm1(y_pred_train)
    ` : `
    y_test_eval = y_test
    y_pred_eval = y_pred_test
    y_train_eval = y_train
    y_train_pred_eval = y_pred_train
    `}

    mae = mean_absolute_error(y_test_eval, y_pred_eval)
    rmse = np.sqrt(mean_squared_error(y_test_eval, y_pred_eval))
    r2 = r2_score(y_test_eval, y_pred_eval)
    mape = np.mean(np.abs((y_test_eval - y_pred_eval) / y_test_eval)) * 100

    results[name] = {
        "model": model,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "y_pred": y_pred_eval
    }

# ==========================================
# 5. MODEL EVALUATION & COMPARISON
# ==========================================
print("\\n--- [5] Model Performance Comparison ---")
comparison_df = pd.DataFrame({
    name: {
        "MAE ($)": f"\${res['MAE']:,.2f}",
        "RMSE ($)": f"\${res['RMSE']:,.2f}",
        "R² Score": f"{res['R2']:.4f}",
        "MAPE (%)": f"{res['MAPE']:.2f}%"
    }
    for name, res in results.items()
}).T

print(comparison_df)

# ==========================================
# 6. K-FOLD CROSS-VALIDATION EVALUATION
# ==========================================
print("\\n--- [6] 5-Fold Cross-Validation (R² Stability) ---")
kf = KFold(n_splits=5, shuffle=True, random_state=${config.randomSeed})
cv_summary = {}

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='r2')
    cv_summary[name] = {
        "CV Mean R²": f"{cv_scores.mean() * 100:.2f}%",
        "CV Std Dev (σ)": f"±{cv_scores.std() * 100:.2f}%",
        "Min Fold R²": f"{cv_scores.min() * 100:.2f}%",
        "Max Fold R²": f"{cv_scores.max() * 100:.2f}%"
    }

print(pd.DataFrame(cv_summary).T)

# ==========================================
# 7. FEATURE IMPORTANCE & VISUALIZATION (Matplotlib)
# ==========================================
rf_model = results["Random Forest"]["model"]
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feature_importance.head(10).plot(kind='barh', color='#3b82f6')
plt.title("Top 10 Feature Importances (Random Forest Regressor)", fontsize=14)
plt.xlabel("Gini / Variance Reduction Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png")
print("\\nSaved feature_importance.png")

# Actual vs. Predicted Plot
plt.figure(figsize=(8, 8))
plt.scatter(y_test_eval, results["Random Forest"]["y_pred"], alpha=0.5, color='#0284c7', label="Predictions")
plt.plot([y_test_eval.min(), y_test_eval.max()], [y_test_eval.min(), y_test_eval.max()], 'r--', lw=2, label="Ideal 45° Line")
plt.xlabel("Actual Sale Price ($)", fontsize=12)
plt.ylabel("Predicted Sale Price ($)", fontsize=12)
plt.title("Actual vs. Predicted House Prices (Random Forest)", fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
print("Saved actual_vs_predicted.png")

plt.show()
`;
}
