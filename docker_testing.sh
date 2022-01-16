#!/bin/bash

# Build docker file
docker build -f Dockerfile . -t trainer:latest

# Run docker file
docker run trainer:latest

