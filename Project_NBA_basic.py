# Project_NBA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sqlite3
import csv
import os
import time
import plotly.express as px
import streamlit as st

db_path = "/Users/anthony/Downloads/Activities/Generation_JDE_Program/JDE_code/data/nba_data/nba.sqlite"  
con = sqlite3.connect(db_path)
data_playersAttributes = pd.read_sql_query("SELECT * FROM Player_Attributes", con)
data_game = pd.read_sql_query("SELECT * FROM Game", con)


####### 1. Total Players who played at least one game 
def total_players():
    query_players = """
    SELECT COUNT(DISTINCT ID)
    FROM Player_Attributes
    """
    df_totalplayers = pd.read_sql_query(query_players, con)
    df_totalplayers.columns = ['Total_Players']
    print(df_totalplayers.to_string(index=False))
total_players()

######## 2. Ratio of Positions
def ratio_positions():
    query_ratiopositions = """
    SELECT POSITION, 
    CAST(COUNT(POSITION) AS float) / 4500 * 100 AS "TOTAL %"
    FROM Player_Attributes
    GROUP BY POSITION
    """
    query_ratio = pd.read_sql_query(query_ratiopositions,con)
    query_ratio = query_ratio[query_ratio['POSITION'].notna() & (query_ratio['POSITION'] != '')]

    plt.bar(query_ratio['POSITION'], query_ratio['TOTAL %'], color='skyblue')
    plt.xlabel('Position')
    plt.ylabel('Total %')
    plt.title('Distribution of Player Positions')
    plt.xticks(rotation=45)
    plt.grid(axis='y')

    plt.tight_layout()
    plt.show()

ratio_positions()

def plot_position_heatmap(data_playersAttributes):
    position_counts = data_playersAttributes['POSITION'].value_counts().reset_index()
    position_counts = position_counts[position_counts['POSITION'].notna() & (position_counts['POSITION'] != '')]
    position_counts.columns = ['Position', 'Count']
    
    # Use pivot_table instead of pivot
    heatmap_data = position_counts.pivot_table(index='Position', values='Count', aggfunc='sum')
    
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Heatmap of Player Positions")
    plt.show()
plot_position_heatmap(data_playersAttributes)

######### 3. Distribution of Height & Weight

data_height = round(data_playersAttributes["HEIGHT"]* 2.54, 1)
distribution_height = px.histogram(
    data_playersAttributes, 
    x=data_height,  
    marginal="box", 
    title="Distribution of Player Heights (in cm)",  
    labels={'x': 'Height (cm)', 'y': 'Count'},
    color_discrete_sequence=["#1f77b4"]  
)
distribution_height.show()

# Box plot for player weights
distribution_weight = px.box(
    data_playersAttributes, 
    x="WEIGHT", 
    title="Box Plot of Player Weights", 
    labels={'x': 'Weight (lbs)', 'y': 'Value'},
    color_discrete_sequence=["#ff7f0e"]
)
distribution_weight.show()

####### 4. Players who players have averaged 20+ point and (5+ rebounds OR 5 assists) per game
def par():
    query = pd.read_sql_query('SELECT DISTINCT DISPLAY_FIRST_LAST, PTS, REB, AST FROM Player_Attributes WHERE PTS > 20 AND (REB > 5 OR AST > 5)', con)
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)  # Disable line wrapping
    print(query)
par()

####### 5. Players who played at least 10 seasons
def players_10():
    players_over10seasons = data_playersAttributes[data_playersAttributes["SEASON_EXP"]>=10]
    players_over10seasons_count = players_over10seasons.shape[0]
    print(players_over10seasons.sort_values(by=["SEASON_EXP"], ascending=False)["DISPLAY_FIRST_LAST"].reset_index(drop=True))
    print(players_over10seasons_count)
players_10()

####### 6. Correlation between Height and Rebound
def corr_height_reb():
    plt.figure(figsize=(10, 6))
    data_playersAttributes['HEIGHT_CM'] = data_playersAttributes['HEIGHT'] * 2.54
    sns.scatterplot(data_playersAttributes, x='HEIGHT_CM', y='REB', alpha=0.6)
    sns.regplot(data_playersAttributes, x='HEIGHT_CM', y='REB', scatter=False, color='red')

    plt.title('Correlation between Height and Rebounds')
    plt.xlabel('Height (cm)')
    plt.ylabel('Rebounds')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
corr_height_reb()

####### 7. Highest Winning Team 
def winninggames_2000_2020():
    query_game = """
    SELECT *
    FROM Game
    """
    df_games = pd.read_sql_query(query_game, con)
    df_games['GAME_DATE'] = pd.to_datetime(df_games['GAME_DATE'], errors='coerce')
    df_official_games = df_games[
            (df_games['GAME_DATE'].dt.year >= 2000) & 
            (df_games['GAME_DATE'].dt.year <= 2020)
    ]
    
    home_wins = df_official_games[df_official_games['WL_HOME'] == 'W'].groupby('TEAM_NAME_HOME').size()
    away_wins = df_official_games[df_official_games['WL_AWAY'] == 'W'].groupby('TEAM_NAME_AWAY').size()


    total_wins = home_wins.add(away_wins, fill_value=0)

    top_teams = total_wins.sort_values(ascending=False).head(30)
    top_teams_df = top_teams.reset_index()
    top_teams_df.columns = ['Team Name', 'Total Wins'] 

    print(top_teams_df.to_string(index=False))

    # Plotting the results
    fig = px.bar(
        top_teams_df,
        x='Total Wins',
        y='Team Name',
        title='Top 30 Teams by Total Wins (2000-2020)',
        labels={'Total Wins': 'Total Wins', 'Team Name': 'Team'},
        color='Total Wins',
        color_continuous_scale=px.colors.sequential.Viridis  # Color scale for better visualization
    )
    
    # Update layout for better readability
    fig.update_layout(
        yaxis_title='Team',
        xaxis_title='Total Wins',
        xaxis=dict(showgrid=True, gridcolor='LightGray'),
        yaxis=dict(showgrid=True, gridcolor='LightGray'),
        template='plotly_white'
    )
    
    fig.show()

winninggames_2000_2020()





