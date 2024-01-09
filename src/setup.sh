#!/bin/bash

# Install requirements from requirements.txt
pip install -r requirements.txt

# Upgrade httpcore and httpx
pip install --upgrade httpcore 
pip install httpx==0.13.3