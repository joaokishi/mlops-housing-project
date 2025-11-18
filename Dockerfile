# 1. Use the official Python image as the base
FROM python:3.9

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file first (for caching)
COPY ./requirements.txt /app/requirements.txt

# 4. Install the libraries
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# 5. Copy the rest of your code (app.py and model.joblib)
COPY . /app

# 6. Start the application
# We use port 7860 because that is what Hugging Face Spaces expects
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]