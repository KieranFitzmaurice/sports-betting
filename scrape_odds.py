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

# Scrape live odds data for each league
leagues = ['NBA','NCAAMB','MLB']

for league in leagues:

    odds_df = sbs.scrape_live_odds(proxypool,league)
    print(f'*** Finished scraping {league} odds ***\n',flush=True)

    if odds_df is not None:

        timestamp = odds_df['observation_datetime'].min()
        timestamp_string = timestamp.replace(microsecond=0).isoformat().replace(':','.')
        outname = os.path.join(pwd,f'data/odds/{league}',timestamp_string + f'_{league}_odds.parquet')
        odds_df.to_parquet(outname)

        print(odds_df.head(),'\n\n',flush=True)

    else:
        print('(No live bets were found)\n\n',flush=True)
