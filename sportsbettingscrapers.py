import numpy as np
import scipy.stats as stats
import scipy.optimize as so
import pandas as pd
import datetime as dt
import requests
import time
import icalendar
import string
import os

# *** Initial setup *** #

def create_folders(leagues=['NBA','NCAAMB','NCAAWB','MLB']):
    """
    Function to create directory structure for scraped data on game odds and outcomes

    param: leagues: list of sports league names
    """

    pwd = os.getcwd()

    folders_to_create = []

    for league in leagues:
        folders_to_create.append(f'data/odds/{league}')
        folders_to_create.append(f'data/outcomes/{league}')
        folders_to_create.append(f'data/prob/{league}')
        folders_to_create.append(f'data/returns/{league}')
        folders_to_create.append(f'data/scores/{league}')
        folders_to_create.append(f'data/schedule/{league}')
        folders_to_create.append(f'data/weights/{league}')

    for folder in folders_to_create:
        folderpath = os.path.join(pwd,folder)
        if not os.path.exists(folderpath):
            os.makedirs(folderpath,exist_ok=True)

    return(None)

# *** Functions to calculate odds-implied probability *** #

def convert_american_to_decimal(american_odds):
    """
    Helper function to convert American odds into decimal odds
    (i.e., dollars paid out per dollar wagered in a winning bet)
    """

    if american_odds < 0: # Negative odds (e.g., bet $110 to make $100)
        decimal_odds = 1 + 100/np.abs(american_odds)

    else: # Positive odds (e.g., bet $100 to make $110)
        decimal_odds = 1 + american_odds/100

    return(decimal_odds)

def convert_decimal_to_american(decimal_odds):
    """
    Helper function to convert decimal odds into American odds
    """

    if decimal_odds >= 2.0:
        american_odds = 100*(decimal_odds-1)
    else:
        american_odds = -100/(decimal_odds-1)

    return(american_odds)

def calculate_implied_probability(odds_data):
    """
    Calculate the implied probability of each event after removing vig using power method.

    See following articles:
    https://researchbank.swinburne.edu.au/file/2069085d-5d5a-4f9c-9f1c-0c52472396cb/1/PDF%20(Published%20version).pdf
    https://pdfs.semanticscholar.org/713d/3cb2e10dec3183ea5feced45bb11097fe702.pdf

    param: odds_data: numpy array of decimal odds. Each row corresponds to a game, and each column corresponds to a team.
    param: p: implied probability after removing vig using power method.
    """

    p = np.zeros(odds_data.shape)

    for i,decimal_odds in enumerate(odds_data):

        p_book = 1/decimal_odds

        f = lambda k: 1 - np.sum(p_book**k)
        fprime = lambda k: -1*np.sum(p_book**k*np.log(p_book))
        fprime2 = lambda k: -1*np.sum(p_book**k*np.log(p_book)**2)

        # Numerically solve for k using Newton's method
        k_guess = 1.0
        k = so.newton(f,k_guess,fprime=fprime,fprime2=fprime2)

        p[i,:] = p_book**k

    return(p)

def calculate_vig(p,q):
    """
    param: p: decimal odds of event occurring
    param: q: decimal odds of event not occurring
    returns: v: calculated vigorish (i.e., return to bookmaker)
    """
    v = 1 - p*q/(p+q)
    return(v)

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


# *** Helper functions to parse data scraped from Action Network and Pinnacle *** #

def parse_actionnetwork(game):
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
    else:
        ht_name = game['teams'][1]['full_name']
        at_name = game['teams'][0]['full_name']

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

def generate_random_string(length=32):
    """
    Helper function to generate a random alphanumeric (base62) string of given length
    """
    alphabet = np.array(list(string.digits + string.ascii_uppercase + string.ascii_lowercase))
    n=len(alphabet)
    random_string = ''.join(alphabet[stats.randint(0,n).rvs(length)])

    return(random_string)

def parse_pinnacle(matchups_results_dict,markets_results_dict):
    """
    param: matchups_results_dict: dictionary of matchup information (e.g., team names, start time) for specific event
    param: markets_results_dict: dictoinary of betting line information for specific event
    """

    # Get information on matchup IDs, which we can use to join betting lines to specific games
    matchups_list = [x for x in matchups_results_dict if x['type']=='matchup']
    matchup_ids = [x['id'] for x in matchups_list]
    markets_list = [x for x in markets_results_dict if x['matchupId'] in matchup_ids and not x['isAlternate']]
    markets_list = [x for x in markets_list if x['period']==0]

    # Create dataframe where each row corresponds to a given game
    matchup_id_list = []
    game_datetime_list = []
    home_team_list = []
    away_team_list = []

    for matchup in matchups_list:

        matchup_id_list.append(matchup['id'])
        game_datetime_list.append(matchup['startTime'])

        if (matchup['participants'][0]['alignment']=='home'):
            home_team_list.append(matchup['participants'][0]['name'])
            away_team_list.append(matchup['participants'][1]['name'])
        else:
            home_team_list.append(matchup['participants'][1]['name'])
            away_team_list.append(matchup['participants'][0]['name'])

    d = {'matchup_id':matchup_id_list,
         'observation_datetime':pd.Timestamp.now(tz='America/New_York'),
         'game_datetime':game_datetime_list,
         'game_date':pd.NA,
         'home_team':home_team_list,
         'away_team':away_team_list,
         'sportsbook_name':'Pinnacle',
         'sportsbook_id':'-1',
         'moneyline_home_odds':pd.NA,
         'moneyline_away_odds':pd.NA,
         'spread_home_value':pd.NA,
         'spread_away_value':pd.NA,
         'spread_home_odds':pd.NA,
         'spread_away_odds':pd.NA,
         'over_value':pd.NA,
         'under_value':pd.NA,
         'over_odds':pd.NA,
         'under_odds':pd.NA}

    odds_df = pd.DataFrame(data=d)
    odds_df['game_datetime'] = pd.to_datetime(odds_df['game_datetime']).dt.tz_convert('America/New_York')
    odds_df['game_date'] = pd.to_datetime(odds_df['game_datetime'].dt.date)

    # Now use matchup ID field to attach betting line information
    odds_df.set_index('matchup_id',inplace=True)

    for market in markets_list:

        index = market['matchupId']
        bet_type = market['type']

        if bet_type == 'moneyline':

            if (market['prices'][0]['designation']=='home'):
                odds_df.loc[index,'moneyline_home_odds'] = market['prices'][0]['price']
                odds_df.loc[index,'moneyline_away_odds'] = market['prices'][1]['price']
            else:
                odds_df.loc[index,'moneyline_home_odds'] = market['prices'][1]['price']
                odds_df.loc[index,'moneyline_away_odds'] = market['prices'][0]['price']

        elif bet_type == 'spread':

            if (market['prices'][0]['designation']=='home'):
                odds_df.loc[index,'spread_home_value'] = market['prices'][0]['points']
                odds_df.loc[index,'spread_home_odds'] = market['prices'][0]['price']
                odds_df.loc[index,'spread_away_value'] = market['prices'][1]['points']
                odds_df.loc[index,'spread_away_odds'] = market['prices'][1]['price']
            else:
                odds_df.loc[index,'spread_home_value'] = market['prices'][1]['points']
                odds_df.loc[index,'spread_home_odds'] = market['prices'][1]['price']
                odds_df.loc[index,'spread_away_value'] = market['prices'][0]['points']
                odds_df.loc[index,'spread_away_odds'] = market['prices'][0]['price']

        elif bet_type == 'total':
            if (market['prices'][0]['designation']=='over'):
                odds_df.loc[index,'over_value'] = market['prices'][0]['points']
                odds_df.loc[index,'over_odds'] = market['prices'][0]['price']
                odds_df.loc[index,'under_value'] = market['prices'][1]['points']
                odds_df.loc[index,'under_odds'] = market['prices'][1]['price']
            else:
                odds_df.loc[index,'over_value'] = market['prices'][1]['points']
                odds_df.loc[index,'over_odds'] = market['prices'][1]['price']
                odds_df.loc[index,'under_value'] = market['prices'][0]['points']
                odds_df.loc[index,'under_odds'] = market['prices'][0]['price']

    odds_df = odds_df.reset_index(drop=True)

    return(odds_df)

# *** Web scraping functions to pull live odds from various sportsbooks *** ###

def get_actionnetwork_url(league,query_date):
    """
    param: league: name of sports league to scrape data for (e.g., 'NBA')
    param: query_date: date of games for which to scrape live odds
    """

    if league == 'NBA':
        url = f'https://api.actionnetwork.com/web/v2/scoreboard/nba?bookIds=15,30,2889,3120,3118,2890,2888,2887,2891,75,123&date={query_date}&periods=event'
    elif league == 'NCAAMB':
        url = f'https://api.actionnetwork.com/web/v2/scoreboard/ncaab?bookIds=15,30,2889,3120,3118,2890,2888,2887,2891,75,123&division=D1&date={query_date}&tournament=0&periods=event'
    elif league == 'NCAAWB':
        url = f'https://api.actionnetwork.com/web/v2/scoreboard/ncaaw?bookIds=15,30,2889,3120,3118,2890,2888,2887,2891,79,75,123&date={query_date}&division=D1&periods=event&tournament=0'
    elif league == 'MLB':
        url = f'https://api.actionnetwork.com/web/v2/scoreboard/mlb?bookIds=15,30,2889,3120,3118,2890,2888,2887,2891,75,123&date={query_date}&periods=event'
    else:
        url = None

    return(url)

def get_pinnacle_url(league):
    """
    param: league: name of sports league to scrape data for (e.g., 'NBA')
    """

    if league == 'NBA':
        matchups_url = 'https://guest.api.arcadia.pinnacle.com/0.1/leagues/487/matchups?brandId=0'
        markets_url = 'https://guest.api.arcadia.pinnacle.com/0.1/leagues/487/markets/straight'
    elif league == 'NCAAMB':
        matchups_url = 'https://guest.api.arcadia.pinnacle.com/0.1/leagues/493/matchups?brandId=0'
        markets_url = 'https://guest.api.arcadia.pinnacle.com/0.1/leagues/493/markets/straight'
    elif league == 'MLB':
        matchups_url = 'https://guest.api.arcadia.pinnacle.com/0.1/leagues/246/matchups?brandId=0'
        markets_url = 'https://guest.api.arcadia.pinnacle.com/0.1/leagues/246/markets/straight'
    else:
        matchups_url = None
        markets_url = None

    return(matchups_url,markets_url)

def scrape_odds_actionnetwork(proxypool,league,sleep_seconds=0.1,random_pause=0.1,days_ahead=3,failure_limit=5):

    """
    Scraper to pull data on live odds from ActionNetwork

    param: proxypool: pool of proxies to route requests through
    param: league: name of sports league to pull odds for
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

        url = get_actionnetwork_url(league,query_date)

        headers = {'Accept': '*/*',
                   'Accept-Encoding': 'gzip, deflate, br',
                   'Accept-Language': 'en-US,en;q=0.9',
                   'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
                  }

        num_failures=0
        success=False

        while num_failures < failure_limit:

            try:
                res = requests.get(url,headers=headers,proxies=proxypool.random_proxy())
                time.sleep(sleep_seconds + dist.rvs())

                if res.ok:
                    results_dict = res.json()
                    observation_datetime = pd.Timestamp.now(tz='America/New_York')
                    success = True
                    break
                else:
                    num_failures += 1
            except Exception as e:
                print(e,flush=True)
                num_failures += 1

        if success:
            if len(results_dict['games']) > 0:

                current_odds_list = []

                for game in results_dict['games']:

                    current_odds_list.append(parse_actionnetwork(game))

                current_odds = pd.concat(current_odds_list)
                current_odds.insert(0, 'observation_datetime', observation_datetime)

                result_list.append(current_odds)

    if len(result_list) > 0:

        odds_df = pd.concat(result_list)

        return odds_df

    else:
        return None

def scrape_odds_pinnacle(proxypool,league,sleep_seconds=0.1,random_pause=0.1,failure_limit=10):

    """
    Scraper to pull data on live MLB odds from pinnacle.com

    param: proxypool: pool of proxies to route requests through
    param: league: name of sports league to pull odds for
    param: sleep_seconds: number of seconds to wait in between api queries
    param: random_pause: total seconds between queries = sleep_seconds + uniform[0,random_pause]
    param: failure_limit: number of times to reattempt scraping if initial request fails
    """

    dist = stats.uniform(0,random_pause)

    matchups_url,markets_url = get_pinnacle_url(league)

    headers = {'accept':'application/json',
               'content-type':'application/json',
               'referer':'https://www.pinnacle.com/',
               'sec-ch-ua':'"Chromium";v="130", "Google Chrome";v="130", "Not?A_Brand";v="99"',
               'sec-ch-ua-mobile':'?0',
               'sec-ch-ua-platform':'"Windows"',
               'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36'}

    num_failures = 0

    # Scrape matchups information
    while num_failures < failure_limit:
        try:
            res = requests.get(url=matchups_url,headers=headers,proxies=proxypool.random_proxy())
            time.sleep(sleep_seconds + dist.rvs())

            if res.ok:
                matchups_results_dict = res.json()
                break
            else:
                num_failures += 1
        except Exception as e:
            print(e,flush=True)
            num_failures += 1

    # Scrape betting line information
    while num_failures < failure_limit:

        try:
            headers['x-api-key'] = generate_random_string(32)
            res = requests.get(url=markets_url,headers=headers,proxies=proxypool.random_proxy())
            time.sleep(sleep_seconds + dist.rvs())

            if res.ok:
                markets_results_dict = res.json()
                break
            else:
                num_failures += 1
        except Exception as e:
            print(e,flush=True)
            num_failures += 1

    # Organize information into dataframe
    try:
        odds_df = parse_pinnacle(matchups_results_dict,markets_results_dict)
        return odds_df
    except:
        return None

def scrape_live_odds(proxypool,league,days_ahead=3):
    """
    Scraper to pull data on live betting odds from various sportsbooks

    param: proxypool: pool of proxies to route requests through
    """

    odds_df_list = []

    # Scrape actionnetwork.com
    odds_df_list.append(scrape_odds_actionnetwork(proxypool,league,days_ahead=days_ahead))

    # Scrape pinnacle.com
    odds_df_list.append(scrape_odds_pinnacle(proxypool,league))

    # Concatenate results
    odds_df_list = [x for x in odds_df_list if x is not None and len(x) > 0]

    if len(odds_df_list) > 0:
        odds_df = pd.concat(odds_df_list)

        # Drop games that have already started
        odds_df = odds_df[odds_df['game_datetime'] > odds_df['observation_datetime']]

        if len(odds_df) > 0:
            odds_df = odds_df.sort_values(by=['game_date','home_team','sportsbook_name']).reset_index(drop=True)
            return odds_df
        else:
            return None

    else:
        return None

# *** National Baseketball Association (NBA) scores and schedule data *** #

def scrape_NBA_scores(proxypool,period=None,start_year=2023,sleep_seconds=0.5):
    """
    Scraper to pull data on live NBA odds from stats.nba.com

    param: proxypool: pool of proxies to route requests through
    param: start_year: Earliest NBA season for which to pull game score information
    param: sleep_seconds: number of seconds to wait in between api queries
    """

    if period is None:
        period = pd.Timestamp.now().to_period('M')

    if period.month < 10:
        season = f'{period.year-1}-{str(period.year)[-2:]}'
    else:
        season = f'{period.year}-{str(period.year+1)[-2:]}'

    season_periods = ['Pre Season', 'Regular Season', 'Playoffs']

    score_df_list = []


    for season_period in season_periods:

        print(season,season_period,flush=True)

        params = {'DateFrom': '',
                  'DateTo': '',
                  'GameSegment': '',
                  'LastNGames': 0,
                  'LeagueID': '00',
                  'Location': '',
                  'MeasureType': 'Base',
                  'Month': 0,
                  'OppTeamID': 0,
                  'Outcome': '',
                  'PORound': 0,
                  'PerMode': 'Totals',
                  'Period': 0,
                  'PlayerID': '',
                  'SeasonSegment': '',
                  'SeasonType': season_period,
                  'Season': season,
                  'ShotClockRange': '',
                  'TeamID': '',
                  'VsConference': '',
                  'VsDivision': ''}

        headers = {'Accept': '*/*',
                   'Accept-Encoding': 'gzip, deflate, br',
                   'Accept-Language': 'en-US,en;q=0.9',
                   'Host': 'stats.nba.com',
                   'Origin': 'https://www.nba.com',
                   'Referer': 'https://www.nba.com/',
                   'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
                  }

        res = requests.get('https://stats.nba.com/stats/teamgamelogs',headers=headers,params=params,proxies=proxypool.random_proxy())
        time.sleep(sleep_seconds)

        results_dict = res.json()

        if len(results_dict['resultSets'][0]['rowSet']) > 0:

            score_df_list.append(extract_NBA_scores(results_dict))

    score_df = pd.concat(score_df_list)
    m = (score_df['game_date'].dt.year == period.year)&(score_df['game_date'].dt.month == period.month)

    if np.sum(m) > 0:
        score_df = score_df[m].reset_index(drop=True)
        return score_df
    else:
        return None

def extract_NBA_scores(results_dict):

    """
    Helper function to process game score information scraped from stats.nba.com
    """

    df = pd.DataFrame(results_dict['resultSets'][0]['rowSet'],columns=results_dict['resultSets'][0]['headers'])

    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df[['GAME_ID','TEAM_ABBREVIATION','TEAM_NAME','GAME_DATE','MATCHUP','WL','PTS']].sort_values(by=['GAME_DATE','GAME_ID'])
    df['HOME_FLAG'] = df['MATCHUP'].apply(lambda x: 'vs.' in x)

    home_df = df[df['HOME_FLAG']][['GAME_ID','TEAM_NAME','TEAM_ABBREVIATION','GAME_DATE','PTS']]
    home_df = home_df.rename(columns={'TEAM_NAME':'home_team','TEAM_ABBREVIATION':'home_abbr','GAME_DATE':'game_date','PTS':'home_score'})

    away_df = df[~df['HOME_FLAG']][['GAME_ID','TEAM_NAME','TEAM_ABBREVIATION','GAME_DATE','PTS']]
    away_df = away_df.rename(columns={'TEAM_NAME':'away_team','TEAM_ABBREVIATION':'away_abbr','GAME_DATE':'game_date','PTS':'away_score'})

    gamelevel_df = pd.merge(home_df,away_df,how='inner',on=['game_date','GAME_ID'])
    gamelevel_df = gamelevel_df.drop(columns='GAME_ID')[['game_date','home_team','away_team','home_abbr','away_abbr','home_score','away_score']]

    return(gamelevel_df)

def scrape_NBA_schedule(proxypool):
    """
    param: proxypool: pool of proxies to route requests through
    """

    url = 'https://ics.ecal.com/ecal-sub/672e7abcc2eaa20008cc96e3/NBA.ics'
    res = requests.get(url,proxies=proxypool.random_proxy())
    cal = icalendar.Calendar.from_ical(res.content)

    game_datetime_list = []
    home_team_list = []
    away_team_list = []

    for component in cal.walk():
        if component.name == 'VEVENT':

            summary = str(component.get('summary'))
            dtstart = component.get('dtstart')

            game_datetime = pd.to_datetime(dtstart.dt).tz_convert('America/New_York')

            try:
                away_team,home_team = summary.split('@')
                home_team = home_team.strip('🏀 ')
                away_team = away_team.strip('🏀 ')

                game_datetime_list.append(game_datetime)
                home_team_list.append(home_team)
                away_team_list.append(away_team)
            except:
                pass

    d = {'game_datetime':game_datetime_list,
         'game_date':pd.NA,
         'home_team':home_team_list,
         'away_team':away_team_list}

    df = pd.DataFrame(data=d)
    df['game_date'] = pd.to_datetime(df['game_datetime'].dt.date)
    df = df.sort_values(by='game_datetime').reset_index(drop=True)

    return(df)


# *** NCAA Division I Mens Basketball scores and schedule data *** #

def scrape_NCAAMB_scores(proxypool,period=None,failure_limit=5,sleep_seconds=0.2):
    """
    param: proxypool: pool of proxies to route requests through
    period: pandas.Period object corresponding to month for which to scrape scores (e.g., 2024-10)
    param: failure_limit: number of times to reattempt scraping if initial request fails
    param: sleep_seconds: number of seconds to wait in between api queries
    """
    if period is None:
        today_date = pd.Timestamp.now()
        year = today_date.year
        month = today_date.month
        period = pd.Period(freq='M',year=year,month=month)

    start_date = period.start_time
    end_date = min(period.end_time,pd.Timestamp.now())
    date_range = pd.date_range(start_date,end_date,freq='D')

    df_list = []

    for date in date_range:

        year = date.year
        month = '{:02}'.format(date.month)
        day = '{:02}'.format(date.day)

        print(date.strftime('%Y-%m-%d'),flush=True)

        url = f'https://data.ncaa.com/casablanca/scoreboard/basketball-men/d1/{year}/{month}/{day}/scoreboard.json'

        headers = {'accept':'application/json, text/javascript, */*; q=0.01',
                   'accept-encoding':'gzip, deflate, br, zstd',
                   'accept-language':'en-US,en;q=0.9',
                   'origin':'https://www.ncaa.com',
                   'priority':'u=1, i',
                   'referer':'https://www.ncaa.com/',
                   'sec-ch-ua':'"Chromium";v="130", "Google Chrome";v="130", "Not?A_Brand";v="99"',
                   'sec-ch-ua-mobile':'?0',
                   'sec-ch-ua-platform':"Windows",
                   'sec-fetch-dest':'empty',
                   'sec-fetch-mode':'cors',
                   'sec-fetch-site':'same-site',
                   'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36'}


        num_failures = 0

        while num_failures < failure_limit:

            try:
                res = requests.get(url,headers=headers,proxies=proxypool.random_proxy())
                time.sleep(sleep_seconds)

                if res.ok:
                    break
                else:
                    num_failures += 1
            except:
                num_failures +=1

        try:
            results_dict = res.json()
            date_scores = extract_NCAAMB_scores(results_dict)

            if date_scores is not None:

                df_list.append(date_scores)

        except:
            pass

    if len(df_list) > 0:

        score_df = pd.concat(df_list).reset_index(drop=True)

        if len(score_df) > 0:
            return score_df
        else:
            return None

    else:
        return None

def extract_NCAAMB_scores(results_dict):
    """
    Helper function to process game score information scraped from data.ncaa.com
    """

    game_date_list = []
    home_team_list = []
    away_team_list = []
    home_abbr_list = []
    away_abbr_list = []
    home_score_list = []
    away_score_list = []

    if list(results_dict.items()) != [('Message','Object not found.')]:

        for i in range(len(results_dict['games'])):

            if results_dict['games'][i]['game']['gameState'] == 'final':

                game_date_list.append(results_dict['games'][i]['game']['startDate'])

                home_team_list.append(results_dict['games'][i]['game']['home']['names']['short'])
                home_abbr_list.append(results_dict['games'][i]['game']['home']['names']['char6'])

                try:
                    home_score_list.append(int(results_dict['games'][i]['game']['home']['score']))
                except:
                    home_score_list.append(pd.NA)


                away_team_list.append(results_dict['games'][i]['game']['away']['names']['short'])
                away_abbr_list.append(results_dict['games'][i]['game']['away']['names']['char6'])

                try:
                    away_score_list.append(int(results_dict['games'][i]['game']['away']['score']))
                except:
                    away_score_list.append(pd.NA)


        d = {'game_date':game_date_list,
             'home_team':home_team_list,
             'away_team':away_team_list,
             'home_abbr':home_abbr_list,
             'away_abbr':away_abbr_list,
             'home_score':home_score_list,
             'away_score':away_score_list}

        df = pd.DataFrame(data=d)
        df[['home_score','away_score']] = df[['home_score','away_score']].astype(int)
        df['game_date'] = pd.to_datetime(df['game_date'])

        return df

    else:

        return None

def scrape_NCAAMB_schedule(proxypool,days_ahead=7,failure_limit=5,sleep_seconds=0.2):
    """
    param: proxypool: pool of proxies to route requests through
    period: pandas.Period object corresponding to month for which to scrape scores (e.g., 2024-10)
    param: failure_limit: number of times to reattempt scraping if initial request fails
    param: sleep_seconds: number of seconds to wait in between api queries
    """

    start_date = pd.Timestamp.now()
    end_date = start_date + pd.Timedelta(days=days_ahead)
    date_range = pd.date_range(start_date,end_date,freq='D')

    df_list = []

    for date in date_range:

        year = date.year
        month = '{:02}'.format(date.month)
        day = '{:02}'.format(date.day)

        print(date.strftime('%Y-%m-%d'),flush=True)

        url = f'https://data.ncaa.com/casablanca/scoreboard/basketball-men/d1/{year}/{month}/{day}/scoreboard.json'

        headers = {'accept':'application/json, text/javascript, */*; q=0.01',
                   'accept-encoding':'gzip, deflate, br, zstd',
                   'accept-language':'en-US,en;q=0.9',
                   'origin':'https://www.ncaa.com',
                   'priority':'u=1, i',
                   'referer':'https://www.ncaa.com/',
                   'sec-ch-ua':'"Chromium";v="130", "Google Chrome";v="130", "Not?A_Brand";v="99"',
                   'sec-ch-ua-mobile':'?0',
                   'sec-ch-ua-platform':"Windows",
                   'sec-fetch-dest':'empty',
                   'sec-fetch-mode':'cors',
                   'sec-fetch-site':'same-site',
                   'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36'}


        num_failures = 0

        while num_failures < failure_limit:

            try:
                res = requests.get(url,headers=headers,proxies=proxypool.random_proxy())
                time.sleep(sleep_seconds)

                if res.ok:
                    break
                else:
                    num_failures += 1
            except:
                num_failures +=1

        try:
            results_dict = res.json()
            date_scores = extract_NCAAMB_schedule(results_dict)

            if date_scores is not None:

                df_list.append(date_scores)

        except:
            pass

    if len(df_list) > 0:

        score_df = pd.concat(df_list).reset_index(drop=True)

        if len(score_df) > 0:
            return score_df
        else:
            return None

    else:
        return None

def extract_NCAAMB_schedule(results_dict):
    """
    Helper function to process game score information scraped from data.ncaa.com
    """

    game_datetime_list = []
    home_team_list = []
    away_team_list = []

    if list(results_dict.items()) != [('Message','Object not found.')]:

        for i in range(len(results_dict['games'])):

            epoch_time = int(results_dict['games'][i]['game']['startTimeEpoch'])
            game_datetime = pd.to_datetime(dt.datetime.fromtimestamp(epoch_time)).tz_localize('America/New_York')

            game_datetime_list.append(game_datetime)
            home_team_list.append(results_dict['games'][i]['game']['home']['names']['short'])
            away_team_list.append(results_dict['games'][i]['game']['away']['names']['short'])

        d = {'game_datetime':game_datetime_list,
             'game_date':pd.NA,
             'home_team':home_team_list,
             'away_team':away_team_list}

        df = pd.DataFrame(data=d)
        df['game_date'] = pd.to_datetime(df['game_datetime'].dt.date)
        df = df.sort_values(by='game_datetime').reset_index(drop=True)

        return df

    else:

        return None
