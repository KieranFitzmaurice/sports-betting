import numpy as np
import pandas as pd
import datetime as dt
import os

def time_since_last_modification(dir):
    """
    This function looks for the most recently modified file in a directory, and
    calculates the time elapsed since it was modified.
    """
    filepaths = [os.path.join(dir,x) for x in os.listdir(dir)]
    timestamps = [os.path.getmtime(x) for x in filepaths]
    modified_time = pd.Timestamp(dt.datetime.fromtimestamp(np.max(timestamps)))
    current_time = pd.Timestamp.now()
    delta_t = current_time - modified_time
    return(delta_t)

def strfdelta(delta_t, fmt):
    d = {"days": delta_t.days}
    hours, rem = divmod(delta_t.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    d["hours"] = '{:02d}'.format(hours)
    d["minutes"] = '{:02d}'.format(minutes)
    d["seconds"] = '{:02d}'.format(seconds)
    return fmt.format(**d)

pwd = os.getcwd()

# Get current date
today_date = pd.Timestamp.now()
yesterday_date = today_date - pd.Timedelta('1D')

today_date_str = today_date.strftime('%Y-%m-%d')
yesterday_date_str = yesterday_date.strftime('%Y-%m-%d')

# Specify leagues to evaluate
leagues = ['NBA','NCAAMB']

with open('report.txt','w') as f:

    f.write(f'###--------------- DAILY REPORT FOR {yesterday_date_str} ---------------###')

    for league in leagues:

        league_str = f'***** {league} *****'
        whitespace = ''.rjust(32-int(len(league_str)/2))
        title_str = f'\n\n{whitespace}{league_str}{whitespace}\n\n'
        f.write(title_str)

        f.write(f'Sportsbook observations:\n\n')

        # Check to see how often a sportsbook has odds available
        sampling_freq = 10 # Number of minutes in between scraping of odds
        samples_per_day = int(24*60/sampling_freq)
        prob_dir = os.path.join(pwd,f'data/prob/{league}')
        filepaths = np.sort([os.path.join(prob_dir,x) for x in os.listdir(prob_dir) if x.startswith(yesterday_date_str)])
        prob_df = pd.concat([pd.read_parquet(filepath) for filepath in filepaths]).reset_index(drop=True)
        prob_df['observation_datetime'] = prob_df['observation_datetime'].dt.floor(f'{sampling_freq}min')

        prob_df = prob_df[['sportsbook_name','observation_datetime']].dropna().drop_duplicates()
        obs_count = prob_df.groupby('sportsbook_name').count()['observation_datetime']
        obs_percent = (100*obs_count/samples_per_day).round(1)

        for book in obs_count.index.values:
            book_string = f'    {book}:'
            obs_string = f'{obs_count.loc[book]} / {samples_per_day} ({obs_percent.loc[book]}%)\n'.rjust(40 - len(book_string))

            f.write(book_string+obs_string)

        # Check to see that data on game scores was recently updated
        dir = os.path.join(pwd,f'data/scores/{league}')
        delta_t = time_since_last_modification(dir)
        delta_t_str = strfdelta(delta_t,'{days} days {hours}h:{minutes}m:{seconds}s')
        f.write(f'\nTime since last update of scores: {delta_t_str}\n')

        # Check to see that on schedule was recently updated
        dir = os.path.join(pwd,f'data/schedule/{league}')
        delta_t = time_since_last_modification(dir)
        delta_t_str = strfdelta(delta_t,'{days} days {hours}h:{minutes}m:{seconds}s')
        f.write(f'Time since last update of schedule: {delta_t_str}\n')
