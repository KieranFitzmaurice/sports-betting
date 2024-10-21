import numpy as np
import scipy.stats as stats
import pandas as pd
import requests
import time
import os

# *** Initial setup *** #

def create_folders(leagues=['NBA','NCAAMB','NCAAWB']):
    """
    Function to create directory structure for scraped data on game odds and outcomes

    param: leagues: list of sports league names
    """

    pwd = os.getcwd()

    folders_to_create = []

    for league in leagues:
        folders_to_create.append(f'data/odds/{league}')
        folders_to_create.append(f'data/scores/{league}')

    for folder in folders_to_create:
        folderpath = os.path.join(pwd,folder)
        if not os.path.exists(folderpath):
            os.makedirs(folderpath,exist_ok=True)

    return(None)

# *** Proxy pool class *** #

class ProxyPool:

    def __init__(self,proxy_list_path):
        """
        param: proxy_list_path: path to list of proxies downloaded form webshare.io
        """
        proxy_list = list(np.loadtxt(proxy_list_path,dtype=str))
        proxy_list = ['http://' + ':'.join(x.split(':')[2:]) + '@' + ':'.join(x.split(':')[:2]) for x in proxy_list]
        proxy_list = [{'http':x,'https':x} for x in proxy_list]

        self.proxy_list = proxy_list
        self.num_proxies = len(self.proxy_list)
        self.random_index = stats.randint(0,self.num_proxies).rvs

    def verify_ip_addresses(self,sleep_seconds=0.1,nmax=10):

        """
        Function to verify that IP address appears as those of proxies
        """
        url = 'https://api.ipify.org/'
        n = np.min([nmax,len(self.proxy_list)])

        for i in range(n):

            res = requests.get(url,proxies=self.proxy_list[i])
            print(res.text,flush=True)
            time.sleep(sleep_seconds)

        return(None)

    def remove_bad_proxies(self,sleep_seconds=0.1):
        """
        Function to remove non-working proxies from list
        """

        url = 'https://api.ipify.org/'
        working_proxies = []

        n_start = self.num_proxies

        for proxy in self.proxy_list:

            try:
                res = requests.get(url,proxies=proxy)
                working_proxies.append(proxy)
            except:
                pass
            time.sleep(sleep_seconds)

        self.proxy_list = working_proxies
        self.num_proxies = len(self.proxy_list)
        self.random_index = stats.randint(0,self.num_proxies).rvs

        n_remove  = n_start - self.num_proxies

        print(f'Removed {n_remove} / {n_start} proxies.',flush=True)

        return(None)


    def random_proxy(self):
        """
        Function to return a randomly-selected proxy server from self.proxy_list
        """

        proxy = self.proxy_list[self.random_index()]

        return(proxy)


# *** Helper functions to parse data scraped from Action Network *** #

def extract_game_information(game):
    """
    param: game: dictionary of odds information for specific event
    """

    # Get start time of match
    start_time = pd.to_datetime(game['start_time']).tz_convert('America/New_York')

    # Get names/abbreviations of each team
    ht_id = game['home_team_id']
    at_id = game['away_team_id']

    if game['teams'][0]['id'] == ht_id:
        ht_name = game['teams'][0]['full_name']
        at_name = game['teams'][1]['full_name']
        ht_abbr = game['teams'][0]['abbr']
        at_abbr = game['teams'][1]['abbr']
    else:
        ht_name = game['teams'][1]['full_name']
        at_name = game['teams'][0]['full_name']
        ht_abbr = game['teams'][1]['abbr']
        at_abbr = game['teams'][0]['abbr']

    book_id_list = []
    book_name_list = []
    moneyline_home_odds_list = []
    moneyline_away_odds_list = []
    spread_home_value_list = []
    spread_away_value_list = []
    spread_home_odds_list = []
    spread_away_odds_list = []
    over_value_list = []
    under_value_list = []
    over_odds_list = []
    under_odds_list = []

    # Dicionary of sportsbook names/ids
    book_id_names = {'15':pd.NA,
                     '30':pd.NA,
                     '75':'BetMGM NJ',
                     '123':'Caesars NJ',
                     '2887':'Fanatics NC',
                     '2888':'FanDuel NC',
                     '2889':'BetMGM NC',
                     '2890':'ESPNBet NC',
                     '3118':'DraftKings NC',
                     '3120':'Caesars NC',
                     '2891':'Bet365 NC'}


    for book_id in game['markets'].keys():

        if book_id in book_id_names.keys():
            book_name = book_id_names[book_id]
        else:
            book_name = pd.NA

        book = game['markets'][book_id]['event']

        # Get information on moneyline bet odds
        try:
            if book['moneyline'][0]['team_id'] == ht_id:
                moneyline_home_odds = book['moneyline'][0]['odds']
                moneyline_away_odds = book['moneyline'][1]['odds']
            else:
                moneyline_home_odds = book['moneyline'][1]['odds']
                moneyline_away_odds = book['moneyline'][0]['odds']
        except:
            moneyline_home_odds = pd.NA
            moneyline_away_odds = pd.NA

        # Get information on spread bet odds
        try:
            if book['spread'][0]['team_id'] == ht_id:
                spread_home_value = book['spread'][0]['value']
                spread_home_odds = book['spread'][0]['odds']
                spread_away_value = book['spread'][1]['value']
                spread_away_odds = book['spread'][1]['odds']

            else:
                spread_home_value = book['spread'][1]['value']
                spread_home_odds = book['spread'][1]['odds']
                spread_away_value = book['spread'][0]['value']
                spread_away_odds = book['spread'][0]['odds']
        except:
            spread_home_value = pd.NA
            spread_home_odds = pd.NA
            spread_away_value = pd.NA
            spread_away_odds = pd.NA


        # Get information on over/under bet odds
        try:
            if book['total'][0]['side'] == 'over':
                over_value = book['total'][0]['value']
                over_odds = book['total'][0]['odds']
                under_value = book['total'][1]['value']
                under_odds = book['total'][1]['odds']
            else:
                over_value = book['total'][1]['value']
                over_odds = book['total'][1]['odds']
                under_value = book['total'][0]['value']
                under_odds = book['total'][0]['odds']
        except:
            over_value = pd.NA
            over_odds = pd.NA
            under_value = pd.NA
            under_odds = pd.NA

        book_id_list.append(book_id)
        book_name_list.append(book_name)
        moneyline_home_odds_list.append(moneyline_home_odds)
        moneyline_away_odds_list.append(moneyline_away_odds)
        spread_home_value_list.append(spread_home_value)
        spread_away_value_list.append(spread_away_value)
        spread_home_odds_list.append(spread_home_odds)
        spread_away_odds_list.append(spread_away_odds)
        over_value_list.append(over_value)
        under_value_list.append(under_value)
        over_odds_list.append(over_odds)
        under_odds_list.append(under_odds)

    data = {'game_datetime':start_time,
            'game_date':pd.NA,
            'home_team':ht_name,
            'away_team':at_name,
            'home_abbr':ht_abbr,
            'away_abbr':at_abbr,
            'sportsbook_name':book_name_list,
            'sportsbook_id':book_id_list,
            'moneyline_home_odds':moneyline_home_odds_list,
            'moneyline_away_odds':moneyline_away_odds_list,
            'spread_home_value':spread_home_value_list,
            'spread_away_value':spread_away_value_list,
            'spread_home_odds':spread_home_odds_list,
            'spread_away_odds':spread_home_odds_list,
            'over_value':over_value_list,
            'under_value':under_value_list,
            'over_odds':over_odds_list,
            'under_odds':under_odds_list}

    df = pd.DataFrame(data)
    df['game_date'] = pd.to_datetime(df['game_datetime'].dt.date)

    return(df)

def harmonize_nba_team_names(x):
    """
    This function converts team names used by sportsbooks to the official names used by the NBA
    """

    name_dict = {'Los Angeles Clippers':'LA Clippers'}

    if x in name_dict.keys():
        return(name_dict[x])
    else:
        return(x)

# *** National Baseketball Association (NBA) Odds *** #

def scrape_NBA_odds(proxypool,sleep_seconds=0.1,random_pause=0.1,days_ahead=30,failure_limit=5):

    """
    Scraper to pull data on live NBA odds from ActionNetwork

    param: proxypool: pool of proxies to route requests through
    param: sleep_seconds: number of seconds to wait in between api queries
    param: random_pause: total seconds between queries = sleep_seconds + uniform[0,random_pause]
    param: days_ahead: number of days in advance to check for newly released odds
    param: failure_limit: number of times to reattempt scraping if initial request fails
    """

    dist = stats.uniform(0,random_pause)

    start_date = pd.Timestamp.now(tz='America/New_York')
    end_date = start_date + pd.Timedelta(days=days_ahead)
    query_dates = [x.strftime('%Y%m%d') for x in pd.date_range(start_date,end_date,freq='D')]

    result_list = []

    for query_date in query_dates:

        url = f'https://api.actionnetwork.com/web/v2/scoreboard/nba?bookIds=15,30,2889,3120,3118,2890,2888,2887,2891,75,123&date={query_date}&periods=event'

        headers = {'Accept': '*/*',
                   'Accept-Encoding': 'gzip, deflate, br',
                   'Accept-Language': 'en-US,en;q=0.9',
                   'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
                  }

        num_failures=0

        while num_failures < failure_limit:

            res = requests.get(url,headers=headers,proxies=proxypool.random_proxy())
            time.sleep(sleep_seconds + dist.rvs())

            if res.ok:
                results_dict = res.json()
                observation_datetime = pd.Timestamp.now(tz='America/New_York')
                success = True
                break
            else:
                num_failures += 1


        if success and len(results_dict['games']) > 0:

            current_odds_list = []

            for game in results_dict['games']:

                current_odds_list.append(extract_game_information(game))

            current_odds = pd.concat(current_odds_list)
            current_odds.insert(0, 'observation_datetime', observation_datetime)

            result_list.append(current_odds)

    if len(result_list) > 0:

        odds_df = pd.concat(result_list)

        # Harmonize team names
        odds_df['home_team'] = odds_df['home_team'].apply(harmonize_nba_team_names)
        odds_df['away_team'] = odds_df['away_team'].apply(harmonize_nba_team_names)

        # Drop games that have already started
        odds_df = odds_df[odds_df['game_datetime'] > odds_df['observation_datetime']].reset_index(drop=True)

        if len(odds_df) > 0:
            return odds_df
        else:
            return None

    else:
        return None

# *** NCAA Division I Mens Basketball Odds *** #

def scrape_NCAAMB_odds(proxypool,sleep_seconds=0.1,random_pause=0.1,days_ahead=30,failure_limit=5):

    """
    Scraper to pull data on live NCAAMB odds from ActionNetwork

    param: proxypool: pool of proxies to route requests through
    param: sleep_seconds: number of seconds to wait in between api queries
    param: random_pause: total seconds between queries = sleep_seconds + uniform[0,random_pause]
    param: days_ahead: number of days in advance to check for newly released odds
    param: failure_limit: number of times to reattempt scraping if initial request fails
    """

    dist = stats.uniform(0,random_pause)

    start_date = pd.Timestamp.now(tz='America/New_York')
    end_date = start_date + pd.Timedelta(days=days_ahead)
    query_dates = [x.strftime('%Y%m%d') for x in pd.date_range(start_date,end_date,freq='D')]

    result_list = []

    for query_date in query_dates:

        url = f'https://api.actionnetwork.com/web/v2/scoreboard/ncaab?bookIds=15,30,2889,3120,3118,2890,2888,2887,2891,75,123&division=D1&date={query_date}&tournament=0&periods=event'

        headers = {'Accept': '*/*',
                   'Accept-Encoding': 'gzip, deflate, br',
                   'Accept-Language': 'en-US,en;q=0.9',
                   'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
                  }

        num_failures=0

        while num_failures < failure_limit:

            res = requests.get(url,headers=headers,proxies=proxypool.random_proxy())
            time.sleep(sleep_seconds + dist.rvs())

            if res.ok:
                results_dict = res.json()
                observation_datetime = pd.Timestamp.now(tz='America/New_York')
                success = True
                break
            else:
                num_failures += 1


        if success and len(results_dict['games']) > 0:

            current_odds_list = []

            for game in results_dict['games']:

                current_odds_list.append(extract_game_information(game))

            current_odds = pd.concat(current_odds_list)
            current_odds.insert(0, 'observation_datetime', observation_datetime)

            result_list.append(current_odds)

    if len(result_list) > 0:

        odds_df = pd.concat(result_list)

        # Drop games that have already started
        odds_df = odds_df[odds_df['game_datetime'] > odds_df['observation_datetime']].reset_index(drop=True)

        if len(odds_df) > 0:
            return odds_df
        else:
            return None

    else:
        return None
