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

# Scrape odds data for NBA
nba_odds_df = sbs.scrape_NBA_odds(proxypool)

timestamp = nba_odds_df['observation_datetime'].min()
timestamp_string = timestamp.replace(microsecond=0).isoformat().replace(':','.')
outname = os.path.join(pwd,'data/odds/NBA',timestamp_string + '_NBA_odds.parquet')
nba_odds_df.to_parquet(outname)

print('*** Finished scraping NBA odds ***\n')
print(nba_odds_df.head(),'\n\n',flush=True)
