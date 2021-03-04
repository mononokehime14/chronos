# Chronos+ Data Science Handover notebook

This repo contains a documented notebook duplicating the data processing, model training and scoring processes in Chronos+ with all the pure engineering parts such as database access and job queueing stripped out.

## Requirements

1. docker
2. docker-compose 
3. Python 3.7.4 (if you are not running locally)

## Run this notebook with docker-compose

`$ make notebook`

The notebook server will be accessible at http://localhost:8080/

## Run this notebook locally

1. Make sure your Python environment is 3.7.4
2. `$ python -r notebook/requirements.txt`
3. `$ jupyter notebook --no-browser --ip=0.0.0.0 --port=8080`

The notebook server will be accessible at http://localhost:8080/
