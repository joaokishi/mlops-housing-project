---
title: Housing Price Predictor
emoji: 🏡
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🏡 End-to-End MLOps: Housing Price Predictor

A production-grade Machine Learning pipeline that automates the training, testing, and deployment of a housing price prediction model.

## 🚀 Live Demo
**API URL:** https://huggingface.co/spaces/joaokishi/mlops-project
**Docs:** https://huggingface.co/spaces/joaokishi/mlops-project/docs

## 🏗 Architecture
This project implements a "Serverless MLOps" architecture (Cardless Stack):
1.  **Training:** Scikit-Learn Linear Regression model trained on California Housing data.
2.  **Tracking:** MLflow (Local) used for experiment tracking and metric logging.
3.  **Serving:** FastAPI microservice wrapped in a Docker container.
4.  **CI/CD:** GitHub Actions pipeline that automatically:
    * Sets up a Python environment.
    * Runs unit tests (`pytest`) to verify model integrity.
    * Deploys the container to Hugging Face Spaces upon success.

## 🛠 Tech Stack
* **Core:** Python 3.9, Scikit-Learn, Pandas
* **API:** FastAPI, Uvicorn
* **DevOps:** Docker, GitHub Actions (CI/CD)
* **Cloud:** Hugging Face Spaces (Hosting)

## 🤖 Automation Logic
Every push to the `main` branch triggers the CI/CD pipeline:
1.  **Test:** Validates that the API responds correctly to sample data.
2.  **Deploy:** If tests pass, the Docker container is rebuilt and pushed to the cloud.
