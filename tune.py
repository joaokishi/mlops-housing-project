import mlflow
import dagshub
import joblib
import random
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# 1. SETUP
dagshub.init(repo_owner='joaokishi', repo_name='mlops-housing-project', mlflow=True)
mlflow.set_tracking_uri(os.environ.get("DAGSHUB_URI"))

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
QUALITY_GATE = 0.65  # Minimum Score required to deploy
EXPERIMENT_NAME = "Housing_Price_Gatekeeper"

mlflow.set_experiment(EXPERIMENT_NAME)

# ---------------------------------------------------------
# TRAIN NEW CANDIDATE
# ---------------------------------------------------------
print("🧪 Training a new candidate model...")

housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, random_state=42)

# Generate Random Config (Simulating active research)
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
        print("Keeping the current production model.")
        mlflow.log_param("status", "rejected")
        
    else:
        # --- ACCEPT ---
        print(f"✅ ACCEPTED: Score ({new_score:.4f}) passed the gate ({QUALITY_GATE}).")
        print("Promoting to Production...")
        mlflow.log_param("status", "deployed")
        
        # Save the model
        joblib.dump(model_pipeline, "model.joblib")
        mlflow.log_artifact("model.joblib")
        
        # Create the flag file so GitHub Actions knows to push
        with open("model_updated.txt", "w") as f:
            f.write("true")