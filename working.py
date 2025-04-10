import requests
import functools
import operator
import pandas as pd
import numpy as np
from pandas import DataFaame, Series

league_id = 34958

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
        f'leagueId={league_id}&scoringPeriod={week}&season=2021')

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

def order_from_schedule(df):
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

if __name__ == '__main__':
    owners = get_teams_in_league(league_id)

    df = get_league_schedule(league_id)
    df[['team1_id', 'team2_id']] = df[['team1_id', 'team2_id']].replace(
        owners.set_index('team_id')['owner_name'].to_dict())

    df_long = schedule_long(df)

    df_long.loc[df_long['week'] == 14, 'team_score'] = np.random.normal(119, 25, size=12)
    df_long.loc[df_long['week'] == 14, 'opp_score'] = np.random.normal(119, 25, size=12)

    order_from_schedule(df_long)
