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

# *** Pull latest NBA game scores *** #
nba_scores_df = sbs.scrape_NBA_scores(proxypool)
print('*** Finished scraping NBA scores ***\n',flush=True)
timestamp_string = nba_scores_df['game_date'].max().strftime('%Y-%m-%d')
outname = os.path.join(pwd,'data/scores/NBA',f'{timestamp_string}_NBA_scores.parquet')
nba_scores_df.to_parquet(outname)
