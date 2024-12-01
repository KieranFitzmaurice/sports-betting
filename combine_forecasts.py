import numpy as np
import pandas as pd
import scipy.interpolate as interp
import sportsbettingscrapers as sbs
import json
import os

### *** BAYESIAN MODEL COMBINATION CLASS *** ###

class ModelCombination:

    def __init__(self,weight_df):
        """
        param: weight_df: dataframe of time-varying model weights
        """

        self.timepoints = np.sort(weight_df.reset_index()['t'].unique())
        self.num_timepoints = len(self.timepoints)

        # Specify function to interpolate between timepoints
        y = np.arange(self.num_timepoints)
        linear_interp = interp.interp1d(self.timepoints,y,kind='linear',bounds_error=False,fill_value='extrapolate')
        self.interp_func = lambda x: np.maximum(np.minimum(linear_interp(x),y.max()),y.min())

        self.models = weight_df.columns.to_list()
        self.num_models = len(self.models)

        # Posterior distribution of model combination weights for each timepoint
        self.posterior_dist = []

        for t in self.timepoints:
            self.posterior_dist.append(weight_df.loc[t].to_numpy())

    def get_timepoint_weights(self,t):
        """
        This function uses linear interpolation to determine how much "time weight" to give to model weights
        estimated at discrete timepoints.

        param: t: continuous time value (scalar)

        returns: it1: index of left discrete timepoint
        returns: it2: index of right discrete timepoint
        returns: wt1: weight given to left discrete timepoint
        returns: wt2: weight given to right discrete timepoint
        """

        it = self.interp_func(t)

        it1 = int(np.floor(it))
        it2 = int(np.ceil(it))

        t1 = self.timepoints[it1]
        t2 = self.timepoints[it2]

        if it1 == it2:
            wt1 = 1.0
            wt2 = 0.0
        else:
            wt2 = (t-t1)/(t2-t1)
            wt1 = 1-wt2

        return it1,it2,wt1,wt2

    def combine_forecasts(self,f_dict,t,alpha=0.05):
        """
        param: f_dict: dictionary of key-value pairs corresponding to the name and forecast of available models
        param: t: continuous time value (scalar)
        param: alpha: significance threshold used to determine credible interval bounds (e.g., 0.05 for 95% CrI)

        returns: f_bar: expected value of combination forcast
        returns: bounds: 100*(1-alpha)% credible interval of combination forcast
        """

        # Values forecasted by each model
        f = np.zeros(self.num_models)

        # Boolean array denoting which models had forecasts available
        v = np.zeros(self.num_models)

        for model_name,forecast_value in f_dict.items():

            if model_name in self.models:

                model_index = self.models.index(model_name)
                f[model_index] = forecast_value
                v[model_index] = 1

        # Get weight associated with each timepoint
        it1,it2,wt1,wt2 = self.get_timepoint_weights(t)

        w1_dist = self.posterior_dist[it1]
        w2_dist = self.posterior_dist[it2]

        num_t1_samples = w1_dist.shape[0]
        num_t2_samples = w2_dist.shape[0]

        t1_sample_weights = wt1*np.ones(num_t1_samples)/num_t1_samples
        t2_sample_weights = wt2*np.ones(num_t2_samples)/num_t2_samples

        f_bar_t1_dist = np.sum(w1_dist*v*f,axis=1)/np.sum(w1_dist*v,axis=1)
        f_bar_t2_dist = np.sum(w2_dist*v*f,axis=1)/np.sum(w2_dist*v,axis=1)

        f_bar_dist = np.concatenate((f_bar_t1_dist,f_bar_t2_dist))
        sample_weights = np.concatenate((t1_sample_weights,t2_sample_weights))

        # Sort from smallest to largest (helpful for calculating CIs)
        sort_inds = np.argsort(f_bar_dist)
        f_bar_dist = f_bar_dist[sort_inds]
        sample_weights = sample_weights[sort_inds]

        # For weighted CDF, Pr(X <= x) = sum(weights[X <= x])
        CDF_vals = np.cumsum(sample_weights)/np.sum(sample_weights)
        PPF_func = interp.interp1d(CDF_vals,f_bar_dist,kind='linear',bounds_error=False,fill_value=(f_bar_dist[0],f_bar_dist[-1]))

        # Get expected value of combination forecast
        f_bar = np.average(f_bar_dist,weights=sample_weights)

        # Get 100*(1-alpha) % credible interval
        LB = float(PPF_func(alpha/2))
        UB = float(PPF_func(1-alpha/2))

        bounds = np.array([LB,UB])

        return f_bar,bounds

    def value_of_weights(self,t_vals=np.arange(-24,0+1,1),alpha=0.05):
        """
        Return the time-varying value of model combination weights

        param: t_vals: timepoints at which to compute model weights
        param: alpha: significance threshold used to determine credible interval bounds (e.g., 0.05 for 95% CrI)

        returns: w_bar_df: dataframe of posterior mean weights
        returns: w_LB_df: lower bound of 100*(1-alpha)% credible interval of weights
        returns: w_UB_df: upper bound of 100*(1-alpha)% credible interval of weights
        """

        w_bar_df = pd.DataFrame({'t':t_vals}).set_index('t')
        w_LB_df = w_bar_df.copy()
        w_UB_df = w_bar_df.copy()

        for model in self.models:

            f_dict = {x:0 for x in self.models}
            f_dict[model] = 1

            w_bar_vals = np.zeros(t_vals.shape)
            w_LB_vals = np.zeros(t_vals.shape)
            w_UB_vals = np.zeros(t_vals.shape)

            for i,t in enumerate(t_vals):

                w_bar,(w_LB,w_UB) = self.combine_forecasts(f_dict,t,alpha=alpha)
                w_bar_vals[i] = w_bar
                w_LB_vals[i] = w_LB
                w_UB_vals[i] = w_UB

            w_bar_df[model] = w_bar_vals
            w_LB_df[model] = w_LB_vals
            w_UB_df[model] = w_UB_vals

        return w_bar_df,w_LB_df,w_UB_df

### *** MAIN *** ###

pwd = os.getcwd()

# Create data folders if they don't already exist
sbs.create_folders()

leagues = ['NBA','NCAAMB']

roi_json_data = []

for league in leagues:

    # Read in implied probabilities
    prob_dir = os.path.join(pwd,f'data/prob/{league}')
    prob_filepath = os.path.join(prob_dir,np.sort(os.listdir(prob_dir))[-1])
    prob_df = pd.read_parquet(prob_filepath)

    # Only keep observations from valid sportsbooks
    prob_df = prob_df[~prob_df['sportsbook_name'].isna()]

    # Round observation timestamp down to nearest 10-minute increment
    prob_df['observation_datetime'] = prob_df['observation_datetime'].dt.floor('10min')

    # Get number of hours until game start time
    prob_df['t'] = (prob_df['observation_datetime'] - prob_df['game_datetime']).dt.total_seconds()/3600

    # Create unique id for each game
    prob_df['matchup'] = prob_df['away_team'] + ' @ ' + prob_df['home_team']
    prob_df['game_id'] = prob_df['game_date'].astype(str) + ' ' + prob_df['matchup']
    game_ids = prob_df['game_id'].unique()

    # Drop duplicate lines (often indicates an issue with parsing team names)
    prob_df.drop_duplicates(subset=['game_id','sportsbook_name','observation_datetime'],keep=False,inplace=True)

    # Read in model combination weights
    weights_dir = os.path.join(pwd,f'data/weights/{league}')
    weights_filepath = os.path.join(weights_dir,np.sort(os.listdir(weights_dir))[-1])
    weights_df = pd.read_parquet(weights_filepath)

    # Initialize model combination class that we'll use to calculate weighted forecasts
    mc = ModelCombination(weights_df)

    roi_df_list = []

    for game_id in game_ids:

        ## Calculate expected ROI for each betting line and save to dataframe

        # Get odds-implied probability from each sportsbook
        m = (prob_df['game_id']==game_id)
        f_dict = prob_df[m][['sportsbook_name','moneyline_home_prob']].set_index('sportsbook_name').to_dict()['moneyline_home_prob']
        t = prob_df[m]['t'].iloc[0]

        # Caclulated weighted-average probability of home win
        f_bar,bounds = mc.combine_forecasts(f_dict,t)

        # Calculate probability of each side of the bet hitting
        home_df = prob_df[m][['observation_datetime','game_datetime','game_id','matchup','sportsbook_name','home_team','moneyline_home_odds']]
        home_df = home_df.rename(columns={'sportsbook_name':'sportsbook','home_team':'side','moneyline_home_odds':'odds'})
        away_df = prob_df[m][['observation_datetime','game_datetime','game_id','matchup','sportsbook_name','away_team','moneyline_away_odds']]
        away_df = away_df.rename(columns={'sportsbook_name':'sportsbook','away_team':'side','moneyline_away_odds':'odds'})
        home_df['hit_prob'] = f_bar
        away_df['hit_prob'] = 1 - f_bar

        # Calculate expected return on investment (ROI)
        game_roi_df = pd.concat([home_df,away_df])
        game_roi_df['EROI'] = game_roi_df['hit_prob']*game_roi_df['odds'] - 1
        roi_df_list.append(game_roi_df)

        # Also save info as JSON file that can be passed to dashboard app
        game_dict = {}
        game_dict['game_datetime'] = home_df['game_datetime'].dt.tz_convert('UTC').iloc[0].isoformat()
        game_dict['observation_datetime'] = home_df['observation_datetime'].dt.tz_convert('UTC').iloc[0].isoformat()
        game_dict['league'] = league
        game_dict['home_team'] = home_df['side'].iloc[0]
        game_dict['away_team'] = away_df['side'].iloc[0]
        game_dict['home_win_prob'] = f_bar
        game_dict['away_win_prob'] = 1 - f_bar

        game_dict['lines'] = game_roi_df[['sportsbook','side','odds','hit_prob','EROI']].to_dict(orient='records')
        roi_json_data.append(game_dict)

    # Save as dataframe
    roi_df = pd.concat(roi_df_list)
    roi_df = roi_df.sort_values(by=['game_datetime','game_id','sportsbook','side']).reset_index(drop=True)

    return_dir = os.path.join(pwd,f'data/returns/{league}/')
    timestamp_str = prob_filepath.split('/')[-1].split('_')[0]

    outname = os.path.join(return_dir,f'{timestamp_str}_{league}_returns.parquet')
    roi_df.to_parquet(outname)

# Save as JSON file
outname = os.path.join(pwd,'sports-betting-dashboard-roi-data.json')
with open(outname,'w') as file:
    json.dump(roi_json_data, file, indent=4)
