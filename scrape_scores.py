import numpy as np
import pandas as pd
import sportsbettingscrapers as sbs
import os

pwd = os.getcwd()

# Create data folders if they don't already exist
sbs.create_folders()

# Initialize proxy pool
proxy_list_path = os.path.join(pwd,'proxies','proxy_list.txt')
proxypool = sbs.ProxyPool(proxy_list_path)

# Get current and last monthly periods
current_period = pd.Timestamp.now().to_period('M')
last_period = current_period-1

for period in [last_period,current_period]:

    period_str = period.strftime('%Y-%m')

    # *** Pull latest data on NBA game scores *** #
    nba_scores_df = sbs.scrape_NBA_scores(proxypool,period=period)

    if nba_scores_df is not None:
        outname = os.path.join(pwd,'data/scores/NBA',f'{period_str}_NBA_scores.parquet')
        nba_scores_df.to_parquet(outname)

    #*** Pull latest data on NCAAMB game scores *** #
    ncaamb_scores_df = sbs.scrape_NCAAMB_scores(proxypool,period=period)

    if ncaamb_scores_df is not None:
        outname = os.path.join(pwd,'data/scores/NCAAMB',f'{period_str}_NCAAMB_scores.parquet')
        ncaamb_scores_df.to_parquet(outname)

    #*** Pull latest data on MLB game scores *** #
    mlb_scores_df = sbs.scrape_MLB_scores(proxypool,period=period)

    if mlb_scores_df is not None:
        outname = os.path.join(pwd,'data/scores/MLB',f'{period_str}_MLB_scores.parquet')
        mlb_scores_df.to_parquet(outname)
