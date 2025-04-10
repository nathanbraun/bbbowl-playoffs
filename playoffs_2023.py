import hosts.db as db
from hosts.league_setup import LEAGUES
import sqlite3
import pandas as pd
from os import path
from pathlib import Path
import seaborn as sns
from textwrap import dedent
from pandas import DataFrame
from utilities import (generate_token, LICENSE_KEY, DB_PATH,
    OUTPUT_PATH, master_player_lookup, schedule_long, get_sims_from_roster)

#####################
# set parameters here
#####################
LEAGUE = 'bb-bowl'
WEEK = 16
WRITE_OUTPUT = False

##############################################
# shouldn't have to change anything below this
##############################################

def summarize_matchup(sims_a, sims_b):
    """
    Given two teams of sims (A and B), summarize a matchup with win
    probability, over-under, betting line, etc
    """

    # start by getting team totals
    total_a = sims_a.sum(axis=1)
    total_b = sims_b.sum(axis=1)

    # get win prob
    winprob_a = (total_a > total_b).mean()
    winprob_b = 1 - winprob_a

    # get over-under
    over_under = (total_a + total_b).median()

    # spread
    spread = (total_a - total_b).median().round(2)
    spread = round(spread*2)/2

    # moneyline
    ml_a = wp_to_ml(winprob_a)

    return {'wp_a': round(winprob_a, 2), 'wp_b': round(winprob_b, 2),
            'over_under': round(over_under, 2), 'spread': spread, 'ml': ml_a}

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

def wp_to_ml(wp):
    if (wp == 0 or wp == 1):
        return 0
    elif wp > 0.5:
        return int(round(-1*(100/((1 - wp)) - 100), 0))
    else:
        return int(round((100/((wp)) - 100), 0))

if __name__ == '__main__':

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

LEAGUE_ID = 34958

player_lookup = master_player_lookup(token)
teams = site.get_teams_in_league(LEAGUE_ID)

nate_id = 217960
mikey_id = 217971

teams['playoffs'] = teams['team_id'].isin([nate_id, mikey_id])

playoff_rosters = pd.concat(
    [site._get_team_roster(x, LEAGUE_ID, player_lookup) for x in
        teams.query('playoffs')['team_id']], ignore_index=True).query('start')
playoff_rosters = playoff_rosters.query("player_id.notnull()")
playoff_rosters['player_id'] = playoff_rosters['player_id'].astype(int)

# add a row to playoff_rosters with value player_position = K, player_id =
# 1999, team_id = 692964

playoff_rosters.loc[56] = Series(
    {'player_position': 'K', 'team_position': 'K', 'team_id': mikey_id,
     'player_id': 1999, 'start': True})

# playoff_rosters = playoff_rosters.query('team_position != "K"')
sims = get_sims_from_roster(token, playoff_rosters, nsims=1000,
                            **league_scoring)

rosters = playoff_rosters

def lineup_by_team(team_id):
    return rosters.query(f"team_id == {team_id} & player_id.notnull()")['player_id']


schedule_this_week = {}
schedule_this_week['team1_id'] = [nate_id]
schedule_this_week['team2_id'] = [mikey_id]

matchup_list = []  # empty matchup list, where all our dicts will go
for a, b in zip(schedule_this_week['team1_id'], schedule_this_week['team2_id']):

    # gives us Series of starting lineups for each team in matchup
    lineup_a = set(lineup_by_team(a)) & set(sims.columns)
    lineup_b = set(lineup_by_team(b)) & set(sims.columns)

    # use lineups to grab right sims, feed into summarize_matchup function
    working_matchup_dict = summarize_matchup(
        sims[list(lineup_a)], sims[list(lineup_b)])

    # add some other info to working_matchup_dict
    working_matchup_dict['team_a'] = a
    working_matchup_dict['team_b'] = b

    # add working dict to list of matchups, then loop around to next
    # matchup
    matchup_list.append(working_matchup_dict)

matchup_df = DataFrame(matchup_list)

team_to_owner = {team: owner for team, owner in zip(teams['team_id'],
                                                    teams['owner_name'])}

matchup_df[['team_a', 'team_b']] = matchup_df[['team_a', 'team_b']].replace(team_to_owner)


team_list = []

for team_id in teams.query('playoffs')['team_id']:
    team_lineup = list(set(lineup_by_team(team_id)) & set(sims.columns))
    working_team_dict = summarize_team(sims[team_lineup])
    working_team_dict['team_id'] = team_id
    working_team_dict['n'] = sims[team_lineup].shape[-1]

    team_list.append(working_team_dict)

team_df = DataFrame(team_list).set_index('team_id')

# high low
# first step: get totals for each team in one DataFrame
totals_by_team = pd.concat(
    [(sims[list(set(lineup_by_team(team_id)) & set(sims.columns))].sum(axis=1)
        .to_frame(team_id)) for team_id in teams.query('playoffs')['team_id']], axis=1)

team_df['p_high'] = (totals_by_team.idxmax(axis=1)
                    .value_counts(normalize=True))

team_df['p_low'] = (totals_by_team.idxmin(axis=1)
                    .value_counts(normalize=True))

# lets see what those high and lows are, on average
# first step: get high score of every sim (max, not idxmax, we don't care
# who got it)
high_score = totals_by_team.max(axis=1)

# same for low score
low_score = totals_by_team.min(axis=1)

# then analyze
pd.concat([
    high_score.describe(percentiles=[.05, .25, .5, .75, .95]).to_frame('high'),
    low_score.describe(percentiles=[.05, .25, .5, .75, .95]).to_frame('low')], axis=1)


# add owner
team_df = (pd.merge(team_df, teams[['team_id', 'owner_name']], left_index=True,
                right_on = 'team_id')
        .set_index('owner_name')
        .drop('team_id', axis=1))

print('\n')
print('**********************************')
print(f'Matchup Projections, Week {WEEK} - 2023')
print('**********************************')
print('\n')
print(matchup_df)
print('\n')
print('**********************************')
print(f'Team Projections, Week {WEEK} - 2023')
print('**********************************')
print('\n')
print(team_df.sort_values('mean', ascending=False).round(2))
