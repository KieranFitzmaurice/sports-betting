import numpy as np
import pandas as pd
import sportsbettingscrapers as sbs
import rapidfuzz as rf
import itertools
import os

### *** Functions for implementing fuzzy-matching *** ###

def distance_func(df,c1,c2):
    """
    Helper function to apply Jaro-Winkler distance calculation to dataframes

    param: df: pandas dataframe
    param: c1: name of column containing left string
    param: c2: name of column containing right string
    """
    f = lambda x: rf.distance.JaroWinkler.distance(x[0],x[1],processor=rf.utils.default_process)
    return df[[c1,c2]].apply(f,axis=1)

def match_team_names(games1,games2):
    """
    Fuzzy matching of team names based on Jaro-Winkler distance metric. This is useful for merging betting
    lines scraped from sportsbooks with box scores scraped from official league websites, which often use
    different naming conventions.

    param: games1: dataframe of matchups scraped from in sportsbook.
    param: games2: dataframe of matchups scraped from official league website.

    Notes:

    Each row of the games1 and games2 dataframes must contain a unique value of "home_team" and "away_team".
    It is assumed that the matchups contained in games1 are a subset of those contained in games2.
    """

    # Get all potential matches of games in dataframe #1 with games in dataframe #2
    potential_matches = pd.DataFrame(itertools.product(games1.index.values,games2.index.values),columns=['index1','index2'])
    potential_matches[['home1','away1']] = games1.loc[potential_matches['index1'],['home_team','away_team']].values
    potential_matches[['home2','away2']] = games2.loc[potential_matches['index2'],['home_team','away_team']].values

    # Calculate Jaro-Winkler distance between matched home and away team names
    home_distance = distance_func(potential_matches,'home1','home2')
    away_distance = distance_func(potential_matches,'away1','away2')
    potential_matches['distance'] = home_distance + away_distance
    potential_matches.sort_values(by='distance',inplace=True)

    matches = []

    # For each iteration, select the closest matching pair of games (as measued by Jaro-Winkler distance)
    # Then drop any remaining potential pairings that include the involved teams.
    # Continue until all games in dataframe #1 are paired up with a game in dataframe #2.

    while len(potential_matches) > 0:

        best_matches = potential_matches.drop_duplicates(['home1','away1'])

        best_match = potential_matches.iloc[0]
        matches.append(best_match.to_numpy())

        m1 = (potential_matches['home1']==best_match['home1'])
        m2 = (potential_matches['home2']==best_match['home2'])
        m3 = (potential_matches['away1']==best_match['away1'])
        m4 = (potential_matches['away2']==best_match['away2'])
        mask = ~(m1|m2|m3|m4)

        potential_matches = potential_matches[mask]

    match_df = pd.DataFrame(matches,columns=potential_matches.columns)
    match_df = match_df[['home1','home2','away1','away2','distance']]

    name_conversion_dict = {}

    part1 = match_df[['home1','home2']].rename(columns={'home1':'name1','home2':'name2'})
    part2 = match_df[['away1','away2']].rename(columns={'away1':'name1','away2':'name2'})

    for index,row in pd.concat([part1,part2]).sort_values(by='name1').iterrows():
        name_conversion_dict[row['name1']] = row['name2']

    return(name_conversion_dict,match_df)

### *** Main *** ###

pwd = os.getcwd()

# Create data folders if they don't already exist
sbs.create_folders()

# Get current time and monthly period
period = pd.Timestamp.now().to_period('M')
period_str = period.strftime('%Y-%m')

# For each league, link data on game outcomes to pre-game betting lines scraped from bookies
leagues = ['NBA','NCAAMB']

for league in leagues:

    print(league,flush=True)

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

    # Game dates and team names from sportsbook
    book_games = prob_df[['game_date','home_team','away_team']].drop_duplicates().reset_index(drop=True)

    # Game dates and team names from official league site
    league_games = score_df[['game_date','home_team','away_team']].drop_duplicates().reset_index(drop=True)

    name_df_list = []

    for date in book_games['game_date'].unique():

        # Get games occurring on specified date
        m1 = (book_games['game_date']==date)
        m2 = (league_games['game_date']==date)

        # Use fuzzy matching to harmonize sportsbook and league naming conventions
        conversion_dict,extra = match_team_names(book_games[m1],league_games[m2])
        conversion_func = lambda x: conversion_dict[x] if x in conversion_dict.keys() else x

        # Convert sportsbook team names to official team names used by league
        m3 = (prob_df['game_date']==date)
        prob_df.loc[m3,'home_team'] = prob_df.loc[m3,'home_team'].apply(conversion_func)
        prob_df.loc[m3,'away_team'] = prob_df.loc[m3,'away_team'].apply(conversion_func)

    # Merge pre-game betting line data with information on final outcome
    prob_df = pd.merge(prob_df,score_df,how='left',on=['game_date','home_team','away_team'])
    prob_df.dropna(subset=['home_score','away_score','moneyline_home_prob','moneyline_away_prob'],inplace=True)

    # Check that probabilities sum to 1.0
    m = np.isclose(prob_df['moneyline_home_prob']+prob_df['moneyline_away_prob'],1.0,atol=0.001)
    prob_df = prob_df[m].reset_index(drop=True)

    # Convert game score into bet outcome (1 if bet hits, 0 otherwise)
    prob_df['moneyline_home_outcome'] = (prob_df['home_score'] > prob_df['away_score']).astype(int)
    prob_df['moneyline_away_outcome'] = 1 - prob_df['moneyline_home_outcome']

    # Save to file
    outname = os.path.join(outcome_dir,f'{period_str}_{league}_outcomes.parquet')
    prob_df.to_parquet(outname)
