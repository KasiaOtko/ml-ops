# Base image
FROM python:3.7-slim

# Install python 
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt install -y wget && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Copy over the application from the computer to the container
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY .dvc/ .dvc/
COPY data.dvc data.dvc
COPY data/ data/
COPY models/ models/
COPY reports/ reports/
COPY .git/ .git/
COPY entrypoint.sh entrypoint.sh

# Set working directory in our container and install dependencies
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

# Name training script as the entrypoint for our docker image
# ENTRYPOINT is the application that we want to run when image is being executed
# ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
ENTRYPOINT ["sh", "entrypoint.sh"]

