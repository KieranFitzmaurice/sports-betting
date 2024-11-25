#!/bin/bash

cd /home/kieran/projects/sports-betting
source sports-betting-venv-1/bin/activate
python scrape_schedules.py
python scrape_scores.py
python attach_outcomes.py
deactivate
mamba activate pymc_env
export OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniforge3/lib
python estimate_weights.py
export LD_LIBRARY_PATH=$OLD_LD_LIBRARY_PATH
unset OLD_LD_LIBRARY_PATH

