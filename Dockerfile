# Use Python 3.11-slim as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN apt-get update && apt-get install -y libgeos-dev
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port Streamlit uses
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "your_app.py", "--server.port=8501", "--server.enableCORS=false"]