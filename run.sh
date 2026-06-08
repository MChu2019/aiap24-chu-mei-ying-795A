#!/bin/bash

echo "🚀 Starting ML Pipeline for AIAP Assessment..."

# Install dependencies
pip install --user -r requirements.txt

# Run the main pipeline
python -m src.main

echo "✅ Pipeline finished successfully!"