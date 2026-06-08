#!/bin/bash
pip install -r requirements.txt
python src/main.pyexport PYTHONPATH="${PYTHONPATH}:$(pwd)"
