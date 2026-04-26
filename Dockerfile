FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app/ app/

# Expose Streamlit's default port
EXPOSE 8501

# Set the default command to run the Streamlit GUI
CMD ["streamlit", "run", "app/src/gui.py", "--server.address", "0.0.0.0"]