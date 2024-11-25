#!/bin/bash

cd ~/projects/sports-betting
source sports-betting-venv-1/bin/activate
python scrape_schedules.py
python scrape_scores.py
python attach_outcomes.py
deactivate
source ~/miniforge3/etc/profile.d/conda.sh
source ~/miniforge3/etc/profile.d/mamba.sh
mamba activate pymc_env
export OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniforge3/lib
python estimate_weights.py
export LD_LIBRARY_PATH=$OLD_LD_LIBRARY_PATH
unset OLD_LD_LIBRARY_PATH
mamba deactivate
