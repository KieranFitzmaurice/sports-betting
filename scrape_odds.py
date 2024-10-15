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

# *** Scrape odds data for NBA *** #
nba_odds_df = sbs.scrape_NBA_odds(proxypool)
print('*** Finished scraping NBA odds ***\n',flush=True)

if nba_odds_df is not None:

    timestamp = nba_odds_df['observation_datetime'].min()
    timestamp_string = timestamp.replace(microsecond=0).isoformat().replace(':','.')
    outname = os.path.join(pwd,'data/odds/NBA',timestamp_string + '_NBA_odds.parquet')
    nba_odds_df.to_parquet(outname)

    print(nba_odds_df.head(),'\n\n',flush=True)

else:
    print('(No live bets were found)\n\n',flush=True)

# *** Scrape odds data for NCAA Division I Mens Basketball *** #
ncaamb_odds_df = sbs.scrape_NCAAMB_odds(proxypool)
print('*** Finished scraping NCAAMB odds ***\n',flush=True)

if ncaamb_odds_df is not None:

    timestamp = ncaamb_odds_df['observation_datetime'].min()
    timestamp_string = timestamp.replace(microsecond=0).isoformat().replace(':','.')
    outname = os.path.join(pwd,'data/odds/NCAAMB',timestamp_string + '_NCAAMB_odds.parquet')
    ncaamb_odds_df.to_parquet(outname)

    print(ncaamb_odds_df.head(),'\n\n',flush=True)
else:
    print('(No live bets were found)\n\n',flush=True)
