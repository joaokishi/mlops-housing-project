---
title: Housing Price Predictor
emoji: 🏡
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🏡 MLOps: Housing Price Predictor

A production-grade Machine Learning system that not only serves predictions but **retrains, tests, and improves itself automatically**.

## 🚀 Live Demo
* **Frontend (Streamlit):** https://mlops-housing-project.streamlit.app/
* **Backend API (Docs):** https://huggingface.co/spaces/joaokishi/mlops-project/docs
* **DagsHub Repo:** https://dagshub.com/joaokishi/mlops-housing-project

## 🏗 Architecture
This project implements a **"Serverless MLOps"** architecture comprising two distinct pipelines:

### 1. The Inference Pipeline (Serving)
* **Model:** Scikit-Learn Random Forest Regressor.
* **API:** FastAPI microservice wrapped in a Docker container.
* **Frontend:** Streamlit dashboard for interactive user predictions.
* **Hosting:** Hugging Face Spaces.

### 2. The Autonomous Training Pipeline (CT)
A self-optimizing loop running on GitHub Actions:
* **Trigger:** Scheduled cron job (Daily) or Manual Dispatch.
* **Experimentation:** A script (`tune.py`) generates random hyperparameters to challenge the current model.
* **Tracking:** All experiments are logged remotely to **DagsHub** using **MLflow**.
* **Quality Gate:** The new model is strictly evaluated.
    * 📉 **Score < 0.65:** Rejected (Deployment skipped).
    * 📈 **Score > 0.65:** Promoted (Artifact updated in Git).

## 🛠 Tech Stack
* **Core:** Python 3.9, Scikit-Learn, Pandas
* **Tracking:** MLflow, DagsHub (Remote Experiment Tracking)
* **API & UI:** FastAPI, Uvicorn, Streamlit
* **DevOps:** Docker, GitHub Actions (CI/CD + Scheduled CT)
* **Cloud:** Hugging Face Spaces

## 🤖 Automation Logic

### A. Continuous Integration (CI/CD)
* **Trigger:** Push to `main` branch.
* **Action:**
    1. Sets up Python environment.
    2. Runs unit tests (`pytest`) to ensure API stability.
    3. If tests pass, builds the Docker container and deploys to production.

### B. Continuous Training (Auto-Tuning)
* **Trigger:** Daily Schedule.
* **Action:**
    1. Trains a "Challenger" model with randomized parameters.
    2. Compares performance against a **Quality Gate (R2 > 0.65)**.
    3. If the Challenger wins, it commits the new `model.joblib` to the repository.
    4. This commit automatically triggers the **CI/CD** pipeline to deploy the new intelligence.
