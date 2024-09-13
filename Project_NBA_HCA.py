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

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Connect to the database
db_path = "/Users/anthony/Downloads/Activities/Generation_JDE_Program/JDE_code/data/nba_data/nba.sqlite"
con = sqlite3.connect(db_path)
data_playersAttributes = pd.read_sql_query("SELECT * FROM Player_Attributes", con)
data_game = pd.read_sql_query("SELECT * FROM Game", con)
data_playersSalary = pd.read_sql_query("SELECT * FROM Player_Salary", con)

game_columns = ['TEAM_ID_HOME', 'TEAM_NAME_HOME','PTS_HOME','WL_HOME', 'TEAM_ID_AWAY', 'TEAM_NAME_AWAY', 'PTS_AWAY','WL_AWAY', 'GAME_DATE']
data_game_columns = data_game[game_columns]

# Verify the null values
# null_values = data_game_columns.isnull().sum()
# print("Null values in each column:")
# print(null_values)

# Remove rows with null values in WL_HOME or WL_AWAY
data_game_columns = data_game_columns.dropna(subset=['WL_HOME', 'WL_AWAY'])

# Verify that the null values have been removed
# null_values_cleaned = data_game_columns.isnull().sum()
# print("\nNull values in each column after cleaning:")
# print(null_values_cleaned)

# --- 1. Combined Points Boxplot for Home and Away Teams ---

# Create a new DataFrame for combined home and away points
points_combined = pd.DataFrame({
    'Points': pd.concat([data_game_columns['PTS_HOME'], data_game_columns['PTS_AWAY']], ignore_index=True), 
    'Team Type': ['Home'] * len(data_game_columns) + ['Away'] * len(data_game_columns)  # Labels for home and away
})

# Plot home and away points as a boxplot
fig_points_box = px.box(points_combined, x='Team Type', y='Points',
                        title='Boxplot of Points (Home vs Away Teams)', 
                        color='Team Type',
                        color_discrete_map={'Home': '#636EFA', 'Away': '#EF553B'})  # Custom colors
fig_points_box.show()


# --- 2. Combined Win-Loss Distribution for Home and Away Teams ---

# Create a new DataFrame for combined win-loss data
wl_combined = pd.DataFrame({
    'Win/Loss': pd.concat([data_game_columns['WL_HOME'], data_game_columns['WL_AWAY']], ignore_index=True),
    'Team Type': ['Home'] * len(data_game_columns) + ['Away'] * len(data_game_columns)  # Labels for home and away
})

# Plot win-loss counts for both home and away teams
fig_wl_combined = px.histogram(wl_combined, x='Win/Loss', color='Team Type',
                               title='Win-Loss Distribution (Home vs Away Teams)',
                               barmode='group',  # Group bars side by side
                               color_discrete_map={'Home': '#636EFA', 'Away': '#EF553B'},  # Custom colors
                               category_orders={'Win/Loss': ['W', 'L']})  # Ensure 'W' comes before 'L'
fig_wl_combined.show()

# --- 3. Scatterplot of Home vs Away Team Points using Plotly ---
fig_scatter = px.scatter(data_game_columns, x='PTS_HOME', y='PTS_AWAY',
                         title='Scatterplot of Home vs Away Team Points',
                         labels={'PTS_HOME': 'Home Team Points', 'PTS_AWAY': 'Away Team Points'},
                         color_discrete_sequence=['#00CC96'])  # Custom color
fig_scatter.show()


# --- 4. User Input
# User input for start and end year
start_year = int(input("Enter start year: "))
end_year = int(input("Enter end year: "))

# Convert GAME_DATE to datetime
data_game_columns['GAME_DATE'] = pd.to_datetime(data_game_columns['GAME_DATE'])

# Extract year and month
data_game_columns['year'] = data_game_columns['GAME_DATE'].dt.year
data_game_columns['month'] = data_game_columns['GAME_DATE'].dt.month

# Filter the data based on user input for start_year and end_year
filtered_data = data_game_columns[(data_game_columns['year'] >= start_year) & (data_game_columns['year'] <= end_year)]

# Calculate monthly win/loss ratios for home and away
monthly_home_wins = filtered_data[filtered_data['WL_HOME'] == 'W'].groupby(['year', 'month']).size()
monthly_home_losses = filtered_data[filtered_data['WL_HOME'] == 'L'].groupby(['year', 'month']).size()
monthly_away_wins = filtered_data[filtered_data['WL_AWAY'] == 'W'].groupby(['year', 'month']).size()
monthly_away_losses = filtered_data[filtered_data['WL_AWAY'] == 'L'].groupby(['year', 'month']).size()

monthly_win_loss = pd.DataFrame({
    'HOME_WINS': monthly_home_wins,
    'HOME_LOSSES': monthly_home_losses,
    'AWAY_WINS': monthly_away_wins,
    'AWAY_LOSSES': monthly_away_losses
}).fillna(0).reset_index()

# Calculate win ratios
monthly_win_loss['HOME_WIN_RATIO'] = monthly_win_loss['HOME_WINS'] / (monthly_win_loss['HOME_WINS'] + monthly_win_loss['HOME_LOSSES'])
monthly_win_loss['AWAY_WIN_RATIO'] = monthly_win_loss['AWAY_WINS'] / (monthly_win_loss['AWAY_WINS'] + monthly_win_loss['AWAY_LOSSES'])

# Calculate average points per month for home and away
monthly_avg_points = filtered_data.groupby(['year', 'month']).agg({
    'PTS_HOME': 'mean',
    'PTS_AWAY': 'mean'
}).reset_index()

# Merge win/loss ratios and average points
monthly_stats = pd.merge(monthly_win_loss, monthly_avg_points, on=['year', 'month'])

# Define the ratio variable here for clarity
ratio = "Win Ratio"

# --- Plot Win/Loss Ratios ---

# Melt the DataFrame for win/loss ratios
win_loss_melted = monthly_stats.melt(id_vars=['year', 'month'], 
                                    value_vars=['HOME_WIN_RATIO', 'AWAY_WIN_RATIO'], 
                                    var_name='Team', value_name=ratio)

# Replace the 'Team' column values with more readable labels
win_loss_melted['Team'] = win_loss_melted['Team'].replace({
    'HOME_WIN_RATIO': 'Home Win Ratio',
    'AWAY_WIN_RATIO': 'Away Win Ratio'
})

# Plot win/loss ratios over time
fig_win_loss = px.line(win_loss_melted, x='month', y=ratio, color='Team', 
                    title=f"Monthly Home and Away Win Ratios ({start_year} - {end_year})",
                    line_shape='linear',
                    labels={'month': 'Month', ratio: 'Win Ratio', 'year': 'Year'},
                    hover_data=['year'],
                    color_discrete_map={'Home Win Ratio': '#636EFA', 'Away Win Ratio': '#EF553B'})

# Show the Win/Loss ratio plot
fig_win_loss.show()


# --- Plot Average Points ---

# Melt the DataFrame for average points
points_melted = monthly_stats.melt(id_vars=['year', 'month'], 
                                value_vars=['PTS_HOME', 'PTS_AWAY'], 
                                var_name='Team', value_name='Points')

# Replace the 'Team' column values with more readable labels
points_melted['Team'] = points_melted['Team'].replace({
    'PTS_HOME': 'Home Team Points',
    'PTS_AWAY': 'Away Team Points'
})

# Plot the average points over time
fig_avg_points = px.line(points_melted, x='month', y='Points', color='Team', 
                        title=f"Monthly Home and Away Average Points ({start_year} - {end_year})",
                        line_shape='linear',
                        labels={'month': 'Month', 'Points': 'Average Points', 'year': 'Year'},
                        hover_data=['year'],
                        color_discrete_map={'Home Team Points': '#636EFA', 'Away Team Points': '#EF553B'})

# Show the Average Points plot
fig_avg_points.show()


######## ---------------------------- MODEL FITTING -----------------------------------------------
# Preprocess the data
data_game_columns['GAME_DATE'] = pd.to_datetime(data_game_columns['GAME_DATE'])
data_game_columns = data_game_columns.dropna(subset=['WL_HOME', 'WL_AWAY'])

# Convert wl_home to binary target variable
data_game_columns['wl_home_binary'] = data_game_columns['WL_HOME'].apply(lambda x: 1 if x == 'W' else 0)

# Select features and target
X = data_game_columns[['PTS_HOME', 'PTS_AWAY']]
y = data_game_columns['wl_home_binary']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print the evaluation results
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
