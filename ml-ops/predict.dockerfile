# Base image
FROM python:3.7-slim

# Install python 
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy over the application from the computer to the container
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY reports/ reports/

# Set working directory in our container and install dependencies
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

# Name training script as the entrypoint for our docker image
# ENTRYPOINT is the application that we want to run when image is being executed
ENTRYPOINT ["python", "-u", "src/models/predict_model.py"]

