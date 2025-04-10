
import hosts.fleaflicker as site
import functools
from hosts.league_setup import LEAGUES
import operator
from pandas import DataFrame, Series
import numpy as np
import requests
import hosts.db as db
import sqlite3
import pandas as pd
from os import path
from pathlib import Path
import seaborn as sns
from textwrap import dedent
from pandas import DataFrame
from utilities import (get_sims, generate_token, LICENSE_KEY, DB_PATH,
                       OUTPUT_PATH, master_player_lookup, get_players,
                       get_sims_from_roster)

def summarize_matchup(sims_a, sims_b):
    """
    Given two teams of sims (A and B), summarize a matchup with win
    probability, over-under, betting line, etc
    """

    # start by getting team totals
    total_a = sims_a.sum(axis=1)
    total_b = sims_b.sum(axis=1)

    # get win prob
    winprob_a = (total_a > total_b).mean().round(2)
    winprob_b = 1 - winprob_a.round(2)

    # get over-under
    over_under = (total_a + total_b).median().round(2)

    # line
    line = (total_a - total_b).median().round(2)
    line = round(line*2)/2

    return {'wp_a': winprob_a, 'wp_b': winprob_b, 'over_under': over_under,
            'line': line}

def summarize_team(sims):
    """
    Calculate summary stats on one set of teams.
    """
    totals = sims.sum(axis=1)
    # note: dropping count, min, max since those aren't that useful
    stats = (totals.describe(percentiles=[.05, .25, .5, .75, .95])
            [['mean', 'std', '5%', '25%', '50%', '75%', '95%']].to_dict())

    # maybe share of points by each pos? commented out now but could look if
    # interesting

    # stats['qb'] = sims.iloc[:,0].mean()
    # stats['rb'] = sims.iloc[:,1:3].sum(axis=1).mean()
    # stats['flex'] = sims.iloc[:,3].mean()
    # stats['wr'] = sims.iloc[:,4:6].sum(axis=1).mean()
    # stats['te'] = sims.iloc[:,6].mean()
    # stats['k'] = sims.iloc[:,7].mean()
    # stats['dst'] = sims.iloc[:,8].mean()

    return stats

def lineup_by_team(team_id):
    return rosters.query(f"team_id == {team_id} & player_id.notnull()")['player_id']

def lock_of_week(df):
    # team a
    wp_a = df[['team_a', 'wp_a', 'team_b']]
    wp_a.columns = ['team', 'wp', 'opp']

    # team b
    wp_b = df[['team_b', 'wp_b', 'team_a']]
    wp_b.columns = ['team', 'wp', 'opp']

    # combine
    stacked = pd.concat([wp_a, wp_b], ignore_index=True)

    # sort highest to low, pick out top
    lock = stacked.sort_values('wp', ascending=False).iloc[0]
    return lock.to_dict()

def photo_finish(df):
    # get the std dev of win probs, lowest will be cloest matchup
    wp_std = df[['wp_a', 'wp_b']].std(axis=1)

    # idxmin "index min" returns the index of the lowest value
    closest_matchup_id = wp_std.idxmin()

    return df.loc[closest_matchup_id].to_dict()

def get_players_from_sims(players, sims):
    players_in_sims = list(set(players) & set(sims))
    return sims[players_in_sims]

def _process_matchup(game):
    return_dict = {}
    return_dict['team1_id'] = game['home']['id']
    return_dict['team2_id'] = game['away']['id']
    return_dict['matchup_id'] = game['id']

    return_dict['team1_score'] = game['homeScore']['score'].get('value')
    return_dict['team2_score'] = game['awayScore']['score'].get('value')

    return return_dict

def _get_schedule_by_week(league_id, week):
    schedule_url = (
        'https://www.fleaflicker.com/api/FetchLeagueScoreboard?' +
        f'leagueId={league_id}&scoringPeriod={week}&season=2024')

    schedule_json = requests.get(schedule_url).json()

    matchup_df = DataFrame([_process_matchup(x) for x in schedule_json['games']])
    matchup_df['season'] = 2024
    matchup_df['week'] = week
    matchup_df['league_id'] = league_id
    return matchup_df

def schedule_long(sched):
    sched1 = sched.rename(columns={'team1_id': 'team_id', 'team2_id':
                                   'opp_id', 'team1_score': 'team_score',
                                   'team2_score': 'opp_score'})
    sched2 = sched.rename(columns={'team2_id': 'team_id', 'team1_id':
                                   'opp_id', 'team2_score': 'team_score',
                                   'team1_score': 'opp_score'})
    return pd.concat([sched1, sched2], ignore_index=True)

def get_league_schedule(league_id):
    return pd.concat([_get_schedule_by_week(league_id, week) for week in
                      range(1, 15)], ignore_index=True)

def get_teams_in_league(league_id):
    teams_url = ('https://www.fleaflicker.com/api/FetchLeagueStandings?' +
                f'leagueId={league_id}')

    teams_json = requests.get(teams_url).json()

    teams_df = _divs_from_league(teams_json['divisions'])
    teams_df['league_id'] = league_id
    teams_df['division'] = [i for i in range(1, 4) for _ in range(4)]
    return teams_df

def _process_team(team):
    dict_to_return = {}

    dict_to_return['team_id'] = team['id']
    dict_to_return['owner_id'] = team['owners'][0]['id']
    dict_to_return['owner_name'] = team['owners'][0]['displayName']

    return dict_to_return

def _teams_from_div(division):
    return DataFrame([_process_team(x) for x in division['teams']])

def _divs_from_league(divisions):
    return pd.concat([_teams_from_div(division) for division in divisions],
                     ignore_index=True)

def matchups_by_teams(df, teams):
    return df.query(f"(team_id in {tuple(teams)}) & (opp_id in {tuple(teams)})")

def return_or_run(_teams, _matchups):
    if len(_teams) == 1:
        return _teams
    else:
        return order_group(_matchups)

def order_group(df, order=[], last=[]):
    team_ids = list(df['team_id'].unique())
    if len(team_ids) == 0:
        return order + last
    elif len(team_ids) == 1:
        return order + team_ids + last
    else:
        # check if team beat everyone
        ave = df.groupby('team_id')[['win', 'lose']].mean()

        winner = list(ave.query("win == 1").index)
        loser = list(ave.query("lose == 1").index)

        if (len(winner) == 1) and (len(loser) == 1):
            new_last = loser + last
            new_df = df.query(f"(team_id != '{loser[0]}') & (team_id != '{winner[0]}')")
            new_order = order + winner
            return order_group(new_df, new_order, new_last)
        elif len(winner) == 1:
            new_df = df.query(f"(team_id != '{winner[0]}')")
            new_order = order + winner
            return order_group(new_df, new_order, last)
        elif len(loser) == 1:
            new_last = loser + last
            new_df = df.query(f"(team_id != '{loser[0]}')")
            return order_group(new_df, order, new_last)
        else:
            best_team = df.loc[df['total_points'].idxmax(), 'team_id']
            new_order =  order + [best_team]
            new_df = df.query(f"(team_id != '{best_team}')")
            return order_group(new_df, new_order, last)

def order_teams(df):
    df['win'] = df['team_score'] > df['opp_score']
    df['lose'] = df['team_score'] < df['opp_score']
    df['tie'] = df['team_score'] == df['opp_score']

    standings = df.groupby('team_id')[['win', 'lose', 'tie']].sum()
    standings['record'] = standings.astype(str).apply('-'.join, axis=1)
    standings.sort_values(['win', 'lose'], ascending=[False, True], inplace=True)

    total_points = df.groupby('team_id').sum()['team_score'].to_frame('total_points')

    df2 = pd.merge(df, total_points, left_on='team_id', right_index=True)

    def _teams_by_record(record):
        return list(standings.query(f"record == '{record}'").index)

    teams = [_teams_by_record(record) for record in standings['record'].unique()]
    matchups = [matchups_by_teams(df2, x) for x in teams]

    return Series(functools.reduce(
        operator.iconcat,
        [return_or_run(x, y) for x, y in zip(teams, matchups)],
        []))

def order_from_schedule(df):
    div_winners = [
        order_teams(df.query(f"division == {i}")).iloc[0] for i in range(1, 4)
    ]
    ordered_non_div_winners = list(order_teams(df.query(f"team_id not in {tuple(div_winners)}")))
    ordered_div_winners = list(order_teams(df.query(f"team_id in {tuple(div_winners)}")))
    return Series(ordered_div_winners + ordered_non_div_winners)


if __name__ == '__main__':
    # set parameters here
    WEEK = 14
    LEAGUE = 'nate-league'

    try:
        assert LEAGUE in LEAGUES.keys()
    except AssertionError:
        print(f"League {LEAGUE} not found. Valid leagues are: {', '.join(list(LEAGUES.keys()))}")


    # first: get league data from DB + roster data by connecting to site
    LEAGUE_ID = LEAGUES[LEAGUE]['league_id']
    conn = sqlite3.connect(DB_PATH)

    teams = db.read_league('teams', LEAGUE_ID, conn)
    schedule = db.read_league('schedule', LEAGUE_ID, conn)
    league = db.read_league('league', LEAGUE_ID, conn)

    # now import site based on host
    host = league.iloc[0]['host']
    if host == 'fleaflicker':
        import hosts.fleaflicker as site
    elif host == 'yahoo':
        import hosts.yahoo as site
    elif host ==  'espn':
        import hosts.espn as site
    elif host ==  'sleeper':
        import hosts.sleeper as site
    else:
        raise ValueError('Unknown host')

    # set other parameters
    host = league.iloc[0]['host']
    league_scoring = {
        'qb':league.iloc[0]['qb_scoring'],
        'skill': league.iloc[0]['skill_scoring'],
        'dst': league.iloc[0]['dst_scoring']
    }

    # then load rosters
    token = generate_token(LICENSE_KEY)['token']
    player_lookup = master_player_lookup(token)

    rosters = site.get_league_rosters(player_lookup, LEAGUE_ID, WEEK)

    # mikey patch
    # mikey_starters = [2088, 904, 172, 547, 720, 990, 1839, 5177]

    # set start = True for player_id in mikey_starters
    # rosters.loc[rosters['player_id'].isin(mikey_starters), 'start'] = True
    rosters = rosters.query("start")

    # and get sims
    sims = get_sims_from_roster(token, rosters, nsims=1000, **league_scoring)

    # making sure we query only valid players
    # available_players = get_players(token, **SCORING)

    # sims = get_sims(token, (set(rosters['fantasymath_id']) &
    #                 set(available_players['fantasymath_id'])),
    #                 nsims=1000, **SCORING)

    # rosters.loc[rosters['name'] == 'Jeff Wilson', 'fantasymath_id'] = 'jeff-wilson'

    # players_w_pts = rosters.query("actual.notnull()")
    # # players_w_pts = pd.read_csv('players.csv')
    # # players_w_pts['skip'] = players_w_pts['skip'].fillna(False)

    # for player, pts in zip(players_w_pts['fantasymath_id'], players_w_pts['actual']):
    #     sims[player] = pts
    # sims['ten-dst'] = 0
    # sims['tee-higgins'] = 0

    owners = get_teams_in_league(LEAGUE_ID)

    df = get_league_schedule(LEAGUE_ID)
    df[['team1_id', 'team2_id']] = df[['team1_id', 'team2_id']].replace(
        owners.set_index('team_id')['owner_name'].to_dict())

    df_long = schedule_long(df)
    df_long = pd.merge(df_long,
                       owners[['owner_name', 'division']],
                       left_on='team_id',
                       right_on='owner_name',
                       )

    # df_long.loc[df_long['week'] == 14, 'team_score'] = np.random.normal(119, 25, size=12)
    # df_long.loc[df_long['week'] == 14, 'opp_score'] = np.random.normal(119, 25, size=12)

    # use sims to get totals by team
    team_totals = pd.concat(
        [get_players_from_sims(lineup_by_team(team_id), sims).sum(axis=1).to_frame(team_id)
         for team_id in teams['team_id']], axis=1)

    team_totals.rename(columns=owners.set_index('team_id')['owner_name'].to_dict(),
                       inplace=True)
    
    # for x in team_totals.columns:
    #     team_totals[x] = np.random.normal(119, 25, size=1000) 

    def sim_14(df_long, sim):
        print(sim.name)
        df14 = df_long.query("week == 14")
        df13 = df_long.query("week < 14")

        df14 = pd.merge(df14.drop('team_score', axis=1),
                        sim.to_frame('team_score'), left_on='team_id',
                        right_index=True)

        df14 = pd.merge(df14.drop('opp_score', axis=1),
                        sim.to_frame('opp_score'), left_on='opp_id',
                        right_index=True)

        return pd.concat([df14, df13])

    week_sims = [order_from_schedule(sim_14(df_long, sim)) for _, sim in team_totals.iterrows()]

    results = pd.concat(week_sims, axis=1).T

    result = results.iloc[0]

    # nate_div = ['strausskahn', 'nbraun', 'pdizz', 'steverynear']
    # ryne_div = ['sporeily', 'kom', 'Ryne', 'craigsoccer99']
    # becker_div = ['Stefense', 'pbecker1313', 'JonHanson', 'BRUZDA']

    # nate_div = ['nbraun', 'komp', 'craigsoccer99', 'Stefense']
    # j_div = ['pdizz', 'Ryne', 'JonHanson', 'strausskahn']
    # oreilly_div = ['Brusda', 'Deising', 'steverynear', 'sporeilly']
    #
    # owners["div"] = [
    #     "nate",
    #     "nate",
    #     "nate",
    #     "nate",
    #     "3j",
    #     "3j",
    #     "3j",
    #     "3j",
    #     "oreilly",
    #     "oreilly",
    #     "oreilly",
    #     "oreilly",
    # ]

    def sum_result(result):
        print(result.name)

        finish = result.reset_index()
        finish.columns = ['order', 'owner_name']
        finish = pd.merge(finish, owners[['owner_name', 'division']])

        division_winners = finish.drop_duplicates('division')
        division_winners['finish'] = ['1 seed', '2 seed', '3 seed']

        non_div_winners = (
            finish.loc[finish.index.difference(division_winners.index)])

        non_div_winners['finish'] = ['wc', '1 pick', '2 pick', '3 pick',
                                     '4 pick', '5 pick', '6 pick', 'duo 11',
                                     'duo 12']

        finish = pd.concat([division_winners, non_div_winners])
        finish['sim'] = result.name
        return finish

    finishes = pd.concat([sum_result(x) for _, x in results.iterrows()])

    final = finishes.groupby(['owner_name'])['finish'].value_counts(normalize=True).to_frame().unstack()

    final.columns = ['1 pick', '1 seed', '2 pick', '2 seed', '3 pick',
                     '3 seed', '4 pick', '5 pick', '6 pick', 'duo 11',
                     'duo 12', 'wc']

    final = final[['1 seed', '2 seed', '3 seed', 'wc', '1 pick', '2 pick',
                   '3 pick', '4 pick', '5 pick', '6 pick', 'duo 11', 'duo 12']]

    final['win_div'] = final[['1 seed', '2 seed', '3 seed']].sum(axis=1)
    final['playoffs'] = final[['win_div', 'wc']].sum(axis=1)
    final['duo'] = final[['duo 11', 'duo 12']].sum(axis=1)

    final.sort_values(['1 seed', '2 seed', '3 seed', 'wc', '1 pick', '2 pick',
                   '3 pick', '4 pick', '5 pick', '6 pick', 'duo 11', 'duo 12'],
                      ascending=[False for _ in range(12)]).round(3)

    fs = finishes.set_index(['sim', 'finish'])[['owner_name']].unstack()
    fs.columns = [x for _, x in fs.columns]

    pairings = fs['duo 11'] + '-' + fs['duo 12']
    most_likely_duos = Series(list(['-'.join(sorted(x)) for x in list(pairings.str.split('-'))]))
    most_likely_duos.value_counts(normalize=True)


