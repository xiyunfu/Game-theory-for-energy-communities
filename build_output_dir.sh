#!/bin/bash

PROJECT_DIRECTORY="/Users/fxy/Game-theory-for-energy-communities/runs"
RUN_DIRECTORY="${PROJECT_DIRECTORY}/$(date +'%Y-%m-%d_%H-%M-%S')"
mkdir -p "$RUN_DIRECTORY"
export RUN_DIRECTORY

# Run the Python script
#/Users/fxy/opt/anaconda3/envs/gametheory/bin/python /Users/fxy/Game-theory-for-energy-communities/main.py
