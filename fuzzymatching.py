import numpy as np
import pandas as pd
import rapidfuzz as rf
import itertools

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

def match_team_names(games1,games2,max_distance=0.67):
    """
    Fuzzy matching of team names based on Jaro-Winkler distance metric. This is useful for merging betting
    lines scraped from sportsbooks with box scores scraped from official league websites, which often use
    different naming conventions.

    param: games1: dataframe of matchups scraped from in sportsbook.
    param: games2: dataframe of matchups scraped from official league website.
    param: max_distance: max combined Jaro-Winkler distance for which to accept a match.

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
  
    max_distance_criteria = (match_df['distance'] <= max_distance)
    unmatch_df = match_df[~max_distance_criteria]
    match_df = match_df[max_distance_criteria]

    name_conversion_dict = {}

    part1 = match_df[['home1','home2']].rename(columns={'home1':'name1','home2':'name2'})
    part2 = match_df[['away1','away2']].rename(columns={'away1':'name1','away2':'name2'})

    for index,row in pd.concat([part1,part2]).sort_values(by='name1').iterrows():
        name_conversion_dict[row['name1']] = row['name2']

    return(name_conversion_dict,match_df,unmatch_df)

def harmonize_team_names(odds_df,schedule_df,max_hours_difference=1.5):
    """
    param: odds_df: dataframe of betting lines data scraped from various sportsbooks
    param: schedule_df: dataframe listing start time, home team, and away team of upcoming games
    param: max_hours_difference: maximum allowed discrepancy in start times listed by sportsbooks versus schedule
    """
    sportsbook_ids = odds_df['sportsbook_id'].unique()
    bad_match_indices = []

    for book_id in sportsbook_ids:

        # Game dates and team names from sportsbook
        m_book = (odds_df['sportsbook_id']==book_id)
        book_games = odds_df[m_book][['game_datetime','home_team','away_team']].drop_duplicates().reset_index(drop=True)

        # Game dates and team names from official league site
        league_games = schedule_df[['game_datetime','home_team','away_team']].drop_duplicates().reset_index(drop=True)

        name_df_list = []

        for datetime in book_games['game_datetime'].unique():

            # Get games occurring on specified date
            m1 = (book_games['game_datetime']==datetime)

            hours_difference = np.abs((league_games['game_datetime'] - datetime).dt.total_seconds()/3600)
            m2 = (hours_difference <= max_hours_difference)

            # Use fuzzy matching to harmonize sportsbook and league naming conventions
            conversion_dict,match_df,unmatch_df = match_team_names(book_games[m1],league_games[m2])
            conversion_func = lambda x: conversion_dict[x] if x in conversion_dict.keys() else x

            if len(unmatch_df) > 0:
                unmatch_df = unmatch_df[['home1','away1']].rename(columns={'home1':'home_team','away1':'away_team'})
                unmatch_df['sportsbook_id'] = book_id
                unmatch_df['game_datetime'] = datetime            
                bad_match_indices += pd.merge(odds_df[unmatch_df.columns].reset_index(),unmatch_df,how='inner',on=list(unmatch_df.columns))['index'].to_list()

            # Convert sportsbook team names to official team names used by league
            m_datetime = (odds_df['game_datetime']==datetime)
            m_conv = m_book&m_datetime
            odds_df.loc[m_conv,'home_team'] = odds_df.loc[m_conv,'home_team'].apply(conversion_func)
            odds_df.loc[m_conv,'away_team'] = odds_df.loc[m_conv,'away_team'].apply(conversion_func)

    # Drop bad matches with poor agreement between sportsbook and official team names
    odds_df = odds_df[~odds_df.index.isin(bad_match_indices)].reset_index(drop=True)

    # Harmonize start times in case there's slight disagreement between books
    # by taking most commonly reported start time for each game 
    start_times = odds_df[['game_date','home_team','away_team','game_datetime']].groupby(['game_date','home_team','away_team']).agg(pd.Series.mode)
    
    odds_df['game_datetime'] = odds_df.apply(lambda x: start_times.loc[x['game_date'],x['home_team'],x['away_team']]['game_datetime'],axis=1)
    odds_df['game_date'] = pd.to_datetime(odds_df['game_datetime'].dt.date)
    
    return(odds_df)


