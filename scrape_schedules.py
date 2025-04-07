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

# Get updated NBA schedule
try:
    nba_schedule = sbs.scrape_NBA_schedule(proxypool)
    outname = os.path.join(pwd,'data/schedule/NBA/NBA_schedule.parquet')
    nba_schedule.to_parquet(outname)
except:
    pass

# Get updated NCAAMB schedule
try:
    ncaamb_schedule = sbs.scrape_NCAAMB_schedule(proxypool)
    outname = os.path.join(pwd,'data/schedule/NCAAMB/NCAAMB_schedule.parquet')
    ncaamb_schedule.to_parquet(outname)
except:
    pass

# Get updated MLB schedule
try:
    mlb_schedule = sbs.scrape_MLB_schedule(proxypool)
    outname = os.path.join(pwd,'data/schedule/MLB/MLB_schedule.parquet')
    mlb_schedule.to_parquet(outname)
except:
    pass
