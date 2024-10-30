#!/bin/bash

cd /home/kieran/projects/sports-betting
source sports-betting-venv-1/bin/activate
python scrape_scores.py
python fuzzy_matching.py
