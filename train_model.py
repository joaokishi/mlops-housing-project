import mlflow
import mlflow.sklearn
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import joblib
from sklearn.pipeline import Pipeline

# 1. PREPARE DATA
print("Loading data...")
housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target)

# 2. TRAIN
print("Training model...")
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42))
])
pipeline.fit(X_train, y_train)
# 3. TRACK WITH MLFLOW
mlflow.set_experiment("No_Cloud_Housing_Project")

with mlflow.start_run():
    score = pipeline.score(X_test, y_test)
    print(f"Model Accuracy: {score}")
    
    mlflow.log_metric("R2_Score", score)
    
    # Save the model locally as a file
    joblib.dump(pipeline, "model.joblib")
    mlflow.log_artifact("model.joblib") # Tracks the file in MLflow

print("Success! Model saved as 'model.joblib'")