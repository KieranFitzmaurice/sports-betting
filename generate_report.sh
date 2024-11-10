#!/bin/bash

cd /home/kieran/projects/sports-betting
source sports-betting-venv-1/bin/activate
python generate_report.py
DATESTR=$(date -d "yesterday 13:00" "+%Y-%m-%d")
SUBJECT="Sportsbook web scraping: summary for $DATESTR"
MESSAGE=$(cat report.txt)
echo "$MESSAGE" | mutt -s "$SUBJECT" kfitzmaurice98@gmail.com,kfitzmaurice@unc.edu,kpf10@pitt.edu

