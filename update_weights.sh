#!/bin/bash

cd /home/kieran/projects/sports-betting
source sports-betting-venv-1/bin/activate
python scrape_schedules.py
python scrape_scores.py
python attach_outcomes.py
