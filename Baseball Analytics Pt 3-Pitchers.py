# Import data to DataFrames
import pandas as pd
# Read in the CSV files
master_df = pd.read_csv('Master.csv',usecols=['playerID','nameFirst','nameLast','bats','throws','debut','finalGame'])
fielding_df = pd.read_csv('Fielding.csv',usecols=['playerID','yearID','stint','teamID','lgID','POS','G','GS','InnOuts','PO','A','E','DP'])
pitching_df = pd.read_csv('Pitching.csv')
awards_df = pd.read_csv('AwardsPlayers.csv', usecols=['playerID','awardID','yearID'])
allstar_df = pd.read_csv('AllstarFull.csv', usecols=['playerID','yearID'])
hof_df = pd.read_csv('HallOfFame.csv',usecols=['playerID','yearid','votedBy','needed_note','inducted','category'])
appearances_df = pd.read_csv('Appearances.csv')
#THIS IS THE ONE I COMMENTED OUT, THUS FIXING THE PROBLEM.

#How dumb, right?

#playoff_pitching_df = pd.read_csv('PitchingPost.csv')
# Print first few rows of `batting_df`
print(pitching_df.head())
print(playoff_pitching_df.head())
# Initialize dictionaries for player stats and years played
player_stats = {}
years_played = {}
# Create dictionaries for player stats and years played from `pitching_df`
for i, row in pitching_df.iterrows():
    playerID = row['playerID']
    if playerID in player_stats:
        player_stats[playerID]['W'] = player_stats[playerID]['W'] + row['W']
        player_stats[playerID]['L'] = player_stats[playerID]['L'] + row['L']
        player_stats[playerID]['G'] = player_stats[playerID]['G'] + row['G']
        player_stats[playerID]['GS'] = player_stats[playerID]['GS'] + row['GS']
        player_stats[playerID]['CG'] = player_stats[playerID]['CG'] + row['CG']
        player_stats[playerID]['SHO'] = player_stats[playerID]['SHO'] + row['SHO']
        player_stats[playerID]['SV'] = player_stats[playerID]['SV'] + row['SV']
        player_stats[playerID]['IPOuts'] = player_stats[playerID]['IPOuts'] + row['IPOuts']
        player_stats[playerID]['H'] = player_stats[playerID]['H'] + row['H']
        player_stats[playerID]['ER'] = player_stats[playerID]['ER'] + row['ER']
        player_stats[playerID]['HR'] = player_stats[playerID]['HR'] + row['HR']
        player_stats[playerID]['BB'] = player_stats[playerID]['BB'] + row['BB']
        player_stats[playerID]['SO'] = player_stats[playerID]['SO'] + row['SO']
        player_stats[playerID]['BAOpp'] = player_stats[playerID]['BAOpp'] + row['BAOpp']
        player_stats[playerID]['ERA'] = player_stats[playerID]['ERA'] + row['ERA']
        player_stats[playerID]['IBB'] = player_stats[playerID]['IBB'] + row['IBB']
        player_stats[playerID]['WP'] = player_stats[playerID]['WP'] + row['WP']
        player_stats[playerID]['HBP'] = player_stats[playerID]['HBP'] + row['HBP']
        player_stats[playerID]['BK'] = player_stats[playerID]['BK'] + row['BK']
        player_stats[playerID]['BFP'] = player_stats[playerID]['BFP'] + row['BFP']
        player_stats[playerID]['GF'] = player_stats[playerID]['GF'] + row['GF']
        player_stats[playerID]['R'] = player_stats[playerID]['R'] + row['R']
        player_stats[playerID]['SH'] = player_stats[playerID]['SH'] + row['SH']
        player_stats[playerID]['SF'] = player_stats[playerID]['SF'] + row['SF']
        player_stats[playerID]['GIDP'] = player_stats[playerID]['GIDP'] + row['GIDP']
        years_played[playerID].append(row['yearID'])
    else:
        player_stats[playerID] = {}
        player_stats[playerID]['W'] = player_stats[playerID]['W'] + row['W']
        player_stats[playerID]['L'] = player_stats[playerID]['L'] + row['L']
        player_stats[playerID]['G'] = player_stats[playerID]['G'] + row['G']
        player_stats[playerID]['GS'] = player_stats[playerID]['GS'] + row['GS']
        player_stats[playerID]['CG'] = player_stats[playerID]['CG'] + row['CG']
        player_stats[playerID]['SHO'] = player_stats[playerID]['SHO'] + row['SHO']
        player_stats[playerID]['SV'] = player_stats[playerID]['SV'] + row['SV']
        player_stats[playerID]['IPOuts'] = player_stats[playerID]['IPOuts'] + row['IPOuts']
        player_stats[playerID]['H'] = player_stats[playerID]['H'] + row['H']
        player_stats[playerID]['ER'] = player_stats[playerID]['ER'] + row['ER']
        player_stats[playerID]['HR'] = player_stats[playerID]['HR'] + row['HR']
        player_stats[playerID]['BB'] = player_stats[playerID]['BB'] + row['BB']
        player_stats[playerID]['SO'] = player_stats[playerID]['SO'] + row['SO']
        player_stats[playerID]['BAOpp'] = player_stats[playerID]['BAOpp'] + row['BAOpp']
        player_stats[playerID]['ERA'] = player_stats[playerID]['ERA'] + row['ERA']
        player_stats[playerID]['IBB'] = player_stats[playerID]['IBB'] + row['IBB']
        player_stats[playerID]['WP'] = player_stats[playerID]['WP'] + row['WP']
        player_stats[playerID]['HBP'] = player_stats[playerID]['HBP'] + row['HBP']
        player_stats[playerID]['BK'] = player_stats[playerID]['BK'] + row['BK']
        player_stats[playerID]['BFP'] = player_stats[playerID]['BFP'] + row['BFP']
        player_stats[playerID]['GF'] = player_stats[playerID]['GF'] + row['GF']
        player_stats[playerID]['R'] = player_stats[playerID]['R'] + row['R']
        player_stats[playerID]['SH'] = player_stats[playerID]['SH'] + row['SH']
        player_stats[playerID]['SF'] = player_stats[playerID]['SF'] + row['SF']
        player_stats[playerID]['GIDP'] = player_stats[playerID]['GIDP'] + row['GIDP']
        years_played[playerID] = []
        years_played[playerID].append(row['yearID'])
