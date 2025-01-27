import numpy as np
import pandas as pd
import pymc as pm
import os

### *** FUNCTIONS *** ###

def preprocess_outcome_data(outcome_df,hours_before=(8,12),n_games=1000,sportsbooks=['DraftKings NC','FanDuel NC','Pinnacle']):
    """
    This function uses a dataframe of pre-game betting lines and post-game outcome data to construct
    numpy arrays encoding information that will be used to parameterize the likelihood function.

    param: outcome_df: pandas dataframe of pre-game betting lines and post-game outcome data
    param: hours_before: number of hours before game start at which to evaluate pre-game betting lines
    param: n_games: include data from the n most recent games
    param: sportsbooks: list of sportsbooks to evaluate (list of length m)

    returns: v: Boolean array denoting whether a betting line was available by game/sportsbook (n x m array)
    returns: f: Odds-implied probability of a home team win by game/sportsbook (n x m array)
    returns: y: Observed post-game outcomes where a value of 1 denotes a home win (vector of length n)
    returns: included_books: names of sportsbooks corresponding to each column of v and f
    """

    # Get forecasts for N most recent games
    recent_games = outcome_df[['game_date','home_team','away_team']].drop_duplicates().sort_values(by='game_date').reset_index(drop=True)
    cutoff_date = recent_games.iloc[-n_games:]['game_date'].min()
    outcome_df = outcome_df[outcome_df['game_date'] >= cutoff_date]

    # Round observation timestamps to nearest 10-minute interval
    outcome_df['observation_datetime'] = outcome_df['observation_datetime'].dt.floor('10min')

    # Get pre-game betting lines at during specified period
    hours_before = np.sort(hours_before)[::-1]
    outcome_df['hours_remaining'] = (outcome_df['observation_datetime'] - outcome_df['game_datetime']).dt.total_seconds()/3600
    m = (outcome_df['hours_remaining'] >= -1*hours_before[0])&(outcome_df['hours_remaining'] < -1*hours_before[1])
    outcome_df = outcome_df[m]

    # Get sportsbooks of interest, and create unique index for each book
    outcome_df = outcome_df[outcome_df['sportsbook_name'].isin(sportsbooks)]
    outcome_df['sportsbook_index'] = outcome_df['sportsbook_name'].apply(lambda x: sportsbooks.index(x))
    outcome_df = outcome_df.sort_values(by=['game_datetime','home_team','sportsbook_index'])

    # Create unique index for each game forecast. In this context, a "forecast" is defined
    # as a prediction (e.g., implied prob of home team win) made by one or more sportsbooks
    # for a specific game at a specific point in time
    forecast_df = outcome_df[['observation_datetime','game_date','home_team','away_team']].drop_duplicates().reset_index(drop=True)
    forecast_df['forecast_index'] = np.arange(len(forecast_df))
    outcome_df = pd.merge(outcome_df,forecast_df,on=['observation_datetime','game_date','home_team','away_team'],how='left')

    # If there are any duplicated forecasts (likely due to incorrect parsing of team names)
    # Drop all duplicates
    outcome_df.drop_duplicates(subset=['forecast_index','sportsbook_index'],keep=False,inplace=True,ignore_index=True)

    # Sportsbook forecast of home win probability
    f_dataframe = outcome_df.pivot(index='forecast_index',columns='sportsbook_index',values='moneyline_home_prob')
    included_books = np.array(sportsbooks)[f_dataframe.columns]
    f = f_dataframe.to_numpy()

    # Boolean array denoting which sportsbooks had forecasts available
    v = 1 - np.isnan(f).astype(int)
    f = np.nan_to_num(f,nan=0)

    # Observed outcome of game (1=home win, 0=loss)
    y = outcome_df[['forecast_index','moneyline_home_outcome']].drop_duplicates()['moneyline_home_outcome'].to_numpy()

    # If only one sportsbook is making a forecast, it doesn't provide any information on the
    # relative predictive performance of each book. So drop rows with only one book present.
    mask = (v.sum(axis=1) > 1)
    f = f[mask]
    v = v[mask]
    y = y[mask]

    return(v,f,y,included_books)

def estimate_combination_weights(v,f,y,draws=2500,tune=1000,n_cores=1):
    """
    param: v: Boolean array denoting whether a betting line was available by game/sportsbook (n x m array)
    param: f: Odds-implied probability of a home team win by game/sportsbook (n x m array)
    param: y: Observed post-game outcomes where a value of 1 denotes a home win (vector of length n)
    param: draws: number of samples to draw from posterior
    param: tune: number of samples to discard from start of chain during burn-in
    param: n_cores: number of available CPU cores

    returns: w_post: posterior distribution of combination weights (n_draws x m array)
    """

    n,m = v.shape

    bmc_model = pm.Model()

    with bmc_model:

        # Specify Dirichlet prior for model weights
        w = pm.Dirichlet('weight',a=np.ones(m))

        # Combination forecast (i.e., expected probability of home team win)
        f_bar = pm.math.sum(w*v*f,axis=1)/pm.math.sum(w*v,axis=1)

        # Define likelihood (sampling distribution) of observations
        y_obs = pm.Bernoulli('y_obs',p=f_bar,observed=y)

        # Sample from posterior
        idata = pm.sample(draws=draws,tune=tune,cores=n_cores)

    # Consolidate results from multiple chains into a single numpy array
    ww = idata.posterior['weight'].to_numpy()
    nchains = ww.shape[0]
    w_post = np.concatenate([ww[i,:,:] for i in range(nchains)],axis=0)

    return(w_post)

### *** MAIN *** ###

pwd = os.getcwd()

# Specify leagues
leagues = ['NBA','NCAAMB']

# Specify number of months to look back for recent game outcomes for each league
# (do this to limit the amount of data we need to read in)
# Note that this period will be shorter for college leagues which play way more games
lookback_months = [12,3]

# Specify sportsbooks to use as "forecasts" when calculating probability of game outcomes
sportsbooks = ['Bet365 NC','BetMGM NJ','Caesars NC','DraftKings NC','ESPNBet NC','FanDuel NC','Pinnacle']

# Specify discrete time intervals leading up to game for which to calculate combination weights
# Units of hours
tmin=0
tmax=24
increment=4

for i,league in enumerate(leagues):

    lookback_period = lookback_months[i]

    # Read in data on past forecasts and game outcomes
    outcome_dir = os.path.join(pwd,f'data/outcomes/{league}')
    outcome_filepaths = [os.path.join(outcome_dir,x) for x in np.sort(os.listdir(outcome_dir))]
    outcome_filepaths = outcome_filepaths[-lookback_period:]

    outcome_df = pd.concat(pd.read_parquet(f) for f in outcome_filepaths).reset_index(drop=True)

    # Calculate model combination weights for each pre-game time interval
    weights_df_list = []

    for t_lower in np.arange(tmin,tmax,increment):

        # Define pre-game time interval of interest
        t_upper = t_lower + increment
        hours_before = (t_lower,t_upper)

        # Estimate combination weights
        v,f,y,included_books = preprocess_outcome_data(outcome_df,hours_before=hours_before,sportsbooks=sportsbooks)
        w_post = estimate_combination_weights(v,f,y)

        # Add to list of weight dataframes
        w_post_df = pd.DataFrame(w_post,columns=included_books)
        w_post_df['t'] = -1*np.mean(hours_before)
        w_post_df = w_post_df.reset_index().rename(columns={'index':'sample'}).set_index(['t','sample'])
        weights_df_list.append(w_post_df)

        # Print update
        print(f'*** {league}: {t_lower}-{t_upper} hours before ***\n')
        print(w_post_df.mean().sort_values(ascending=False),'\n')

    # Concatenate dataframes for each time interval
    weights_df = pd.concat(weights_df_list).fillna(0)

    # Save results
    date_str = pd.Timestamp.now().strftime('%Y-%m-%d')
    outname = os.path.join(pwd,f'data/weights/{league}/{date_str}_{league}_weights.parquet')
    weights_df.to_parquet(outname)
