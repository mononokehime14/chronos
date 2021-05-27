# Chronos+ Data Science Handover notebook

This repo contains a documented notebook duplicating the data processing, model training and scoring processes in Chronos+ with all the pure engineering parts such as database access and job queueing stripped out. The notebook is at `notebook/Chronos+ Handover.ipynb`.


## Run this notebook locally with docker-compose

### Requirements

1. docker
2. docker-compose 

### Run the notebook

`$ make container`

The notebook server will be accessible at http://localhost:8080/

## Run this notebook locally without docker-compose

### Requirements

1. Python 3.7.4

### Run the notebook

1. `make local`

The notebook server will be accessible at http://localhost:8080/

## Run this notebook on some remote Jupyter server

### Requirements

1. Notebook server is being run on Python 3.7.4
2. Connect to the server
3. Upload the `notebook` directory
4. Install the requirements in `notebook/requirements.txt`

## Run Pipeline notebook

### Run Training PyTorch Autoencoder model

1. notebook/pytorch_model_training.ipynb

### Run Benchmark Testing

1. notebook/pytorch_model_benchmark.ipynb