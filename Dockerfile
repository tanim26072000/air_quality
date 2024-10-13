# Use Python 3.11-slim as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Clone your GitHub repository into the working directory
RUN apt-get update && apt-get install -y git && \
    git clone https://github.com/tanim26072000/air_quality.git /app

# Install required system packages (e.g., for geopandas)
RUN apt-get install -y libgeos-dev

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the Streamlit default port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
