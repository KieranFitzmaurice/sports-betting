import numpy as np
import pandas as pd
import sportsbettingscrapers as sbs
import scipy.optimize as so
import os

pwd = os.getcwd()

# Create data folders if they don't already exist
sbs.create_folders()

leagues = ['NBA','NCAAMB']

for league in leagues:

    odds_dir = os.path.join(pwd,f'data/odds/{league}')
    prob_dir = os.path.join(pwd,f'data/prob/{league}')

    odds_timestamps = np.sort([x.split('_')[0] for x in os.listdir(odds_dir)])
    prob_timestamps = np.sort([x.split('_')[0] for x in os.listdir(prob_dir)])

    # Get timestamps of odds that we haven't yet converted to implied probabilities
    new_timestamps = odds_timestamps[~np.isin(odds_timestamps,prob_timestamps)]

    for timestamp_str in new_timestamps:

        odds_filepath = os.path.join(odds_dir,f'{timestamp_str}_{league}_odds.parquet')
        prob_filepath = os.path.join(prob_dir,f'{timestamp_str}_{league}_prob.parquet')

        odds_df = pd.read_parquet(odds_filepath)

        # Since focusing on moneyline bets, can drop columns corresponding to spread/total bets
        odds_df = odds_df.iloc[:,:11]

        # Convert american odds to decimal odds
        odds_df['moneyline_home_odds'] = odds_df['moneyline_home_odds'].apply(sbs.convert_american_to_decimal)
        odds_df['moneyline_away_odds'] = odds_df['moneyline_away_odds'].apply(sbs.convert_american_to_decimal)

        # Calculate odds-implied probability
        prob_df = odds_df.copy()
        prob_df[~prob_df[['moneyline_home_odds','moneyline_away_odds']].isna().any(axis=1)]
        prob_df[['moneyline_home_prob','moneyline_away_prob']] = sbs.calculate_implied_probability(prob_df[['moneyline_home_odds','moneyline_away_odds']].to_numpy())

        # Save to file
        prob_df.to_parquet(prob_filepath)
