import mlflow
import joblib
import random
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# 1. SETUP (Pure MLflow)
# We removed dagshub.init(). We rely on environment variables passed by GitHub Actions.
tracking_uri = os.environ.get("DAGSHUB_URI")
if not tracking_uri:
    print("⚠️ Warning: DAGSHUB_URI not found. Using local tracking.")
else:
    mlflow.set_tracking_uri(tracking_uri)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
QUALITY_GATE = 0.65
EXPERIMENT_NAME = "Housing_Price_Gatekeeper"

mlflow.set_experiment(EXPERIMENT_NAME)

# ---------------------------------------------------------
# TRAIN NEW CANDIDATE
# ---------------------------------------------------------
print("🧪 Training a new candidate model...")

housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, random_state=42)

# Generate Random Config
n_estimators = random.randint(10, 200)
max_depth = random.randint(2, 20)

model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42))
])

model_pipeline.fit(X_train, y_train)
new_score = model_pipeline.score(X_test, y_test)

print(f"📊 Candidate Score: {new_score:.4f}")

# ---------------------------------------------------------
# THE QUALITY GATE LOGIC
# ---------------------------------------------------------
with mlflow.start_run():
    mlflow.log_metric("R2_Score", new_score)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    
    if new_score < QUALITY_GATE:
        # --- REJECT ---
        print(f"❌ REJECTED: Score ({new_score:.4f}) is below threshold ({QUALITY_GATE}).")
        mlflow.log_param("status", "rejected")
        
    else:
        # --- ACCEPT ---
        print(f"✅ ACCEPTED: Score ({new_score:.4f}) passed the gate ({QUALITY_GATE}).")
        mlflow.log_param("status", "deployed")
        
        # Save model
        joblib.dump(model_pipeline, "model.joblib")
        mlflow.log_artifact("model.joblib")
        
        # Create flag file for GitHub Actions
        with open("model_updated.txt", "w") as f:
            f.write("true")