#!/bin/bash

# Install requirements from requirements.txt
pip install --upgrade pip
pip install -r requirements.txt
pip install --upgrade httpcore httpx googletrans==4.0.0-rc1
