FROM python:3.11-slim

WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY app/ ./app/

# By default, run the CLI pipeline. 
# We could also expose streamlit, but the requirements just say "application container".
CMD ["python", "app/src/main.py"]
