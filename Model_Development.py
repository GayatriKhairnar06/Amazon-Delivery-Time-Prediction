import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import math

# ======================
# Load the processed dataset
# ======================
df = pd.read_csv("amazon_processed.csv")

# Drop target leakage columns
if "Order_ID" in df.columns:
    df = df.drop("Order_ID", axis=1)

# Fill any remaining missing numeric values with median
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill any remaining categorical missing values with mode
categorical_cols = df.select_dtypes(include='object').columns.tolist()
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ======================
# Define features and target
# ======================
X = df.drop("Delivery_Time", axis=1)
y = df["Delivery_Time"]

# One-hot encoding for categorical features
categorical_cols = X.select_dtypes(include='object').columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)],
    remainder="passthrough"
)

# ======================
# Define models
# ======================
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
#tracking variables
best_model = None
best_model_name = None
best_r2 = -float("inf")

# ======================
# Training and Evaluation with MLflow
# ======================
for model_name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    with mlflow.start_run(run_name=model_name):
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Log metrics
        mlflow.log_param("model", model_name)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        if r2 > best_r2:
            best_r2 = r2
            best_model = pipeline
            best_model_name = model_name

        # Log model
        mlflow.sklearn.log_model(pipeline, model_name)

        # Print summary
        print(f"{model_name} Performance:")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R2: {r2:.2f}")
        print("=" * 40)
print(f"\nBest model selected: {best_model_name} with R2 = {best_r2:.2f}")

# Save the best model
mlflow.sklearn.log_model(best_model, "Best_Model")

import joblib

# Save best model locally
joblib.dump(best_model, "best_model.pkl")
print("Best model saved as best_model.pkl")

