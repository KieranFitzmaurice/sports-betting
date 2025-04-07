import numpy as np
import pandas as pd
import sportsbettingscrapers as sbs
import fuzzymatching as fm
import os

pwd = os.getcwd()

# Create data folders if they don't already exist
sbs.create_folders()

# Specify leagues to evaluate
leagues = ['NBA','NCAAMB','MLB']

# Get current time and monthly period
current_period = pd.Timestamp.now().to_period('M')
last_period = current_period-1

for period in [last_period,current_period]:

    period_str = period.strftime('%Y-%m')

    # For each league, link data on game outcomes to pre-game betting lines scraped from bookies

    for league in leagues:

        print(period_str,league,flush=True)

        prob_dir = os.path.join(pwd,f'data/prob/{league}')
        score_dir = os.path.join(pwd,f'data/scores/{league}')
        outcome_dir = os.path.join(pwd,f'data/outcomes/{league}')

        # Read in data on betting lines scraped from sportsbooks for current month
        prob_filenames = np.sort([x for x in os.listdir(prob_dir) if x.startswith(period_str)])

        # If there's no data to process, terminate early and skip to next league
        if len(prob_filenames) == 0:
            continue

        prob_filepaths = [os.path.join(prob_dir,x) for x in prob_filenames]
        prob_df = pd.concat([pd.read_parquet(x) for x in prob_filepaths]).reset_index(drop=True)

        # Read in data on game scores scraped from league websites for current month
        score_filepath = os.path.join(score_dir,f'{period_str}_{league}_scores.parquet')
        score_df = pd.read_parquet(score_filepath).drop(columns=['home_abbr','away_abbr'])

        # Only keep lines for games that have been completed
        prob_df = prob_df[prob_df['game_date'] <= score_df['game_date'].max()]

        ## Harmonize team names

        # Score data doesn't always include game start time
        # So set the game_datetime column to midnight arbitrarily for matching purposes
        # We'll later set it back to the true value
        prob_df['game_datetime_true'] = prob_df['game_datetime'].copy()
        prob_df['game_datetime'] = prob_df['game_date'].copy()
        score_df['game_datetime'] = score_df['game_date'].copy()

        prob_df = fm.harmonize_team_names(prob_df,score_df)

        prob_df['game_datetime'] = prob_df['game_datetime_true']
        score_df.drop(columns='game_datetime',inplace=True)

        # Merge pre-game betting line data with information on final outcome
        prob_df = pd.merge(prob_df,score_df,how='left',on=['game_date','home_team','away_team'])
        prob_df.dropna(subset=['home_score','away_score','moneyline_home_prob','moneyline_away_prob'],inplace=True)

        keepcols = ['observation_datetime',
                    'game_datetime',
                    'game_date',
                    'home_team',
                    'away_team',
                    'sportsbook_name',
                    'sportsbook_id',
                    'moneyline_home_odds',
                    'moneyline_away_odds',
                    'moneyline_home_prob',
                    'moneyline_away_prob',
                    'home_score',
                    'away_score']

        prob_df = prob_df[keepcols]

        # Check that probabilities sum to 1.0
        m = np.isclose(prob_df['moneyline_home_prob']+prob_df['moneyline_away_prob'],1.0,atol=0.001)
        prob_df = prob_df[m].reset_index(drop=True)

        # Convert game score into bet outcome (1 if bet hits, 0 otherwise)
        prob_df['moneyline_home_outcome'] = (prob_df['home_score'] > prob_df['away_score']).astype(int)
        prob_df['moneyline_away_outcome'] = 1 - prob_df['moneyline_home_outcome']

        # Save to file
        outname = os.path.join(outcome_dir,f'{period_str}_{league}_outcomes.parquet')
        prob_df.to_parquet(outname)
