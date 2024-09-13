# README.md for NBA Project

## PROJECT OVERVIEW

**NBA Project** is an analytical tool designed for basketball enthusiasts and data analysts to explore and visualize NBA player and game statistics. This project utilizes SQLite databases to extract meaningful insights from the data, enabling users to understand player performance, team dynamics, and game outcomes effectively. The project consists of two main scripts: `Project_NBA_basic.py` and `Project_NBA_HCA.py`, each serving distinct analytical purposes.

## TABLE OF CONTENTS

- [Purpose](#purpose)
- [Installation](#installation)
- [Usage](#usage)
- [Data Visualization](#data-visualization)
- [Conclusion](#conclusion)

## PURPOSE

The primary objective of this project is to provide a comprehensive analysis of NBA data through various statistical methods and visualizations. Key functionalities include:

- **Player Analysis**: Examine player attributes, including height, weight, and scoring averages.
- **Team Performance**: Analyze win-loss records and team statistics over time.
- **Predictive Modeling**: Utilize machine learning techniques to predict game outcomes based on historical data.

This project is significant as it not only aids in understanding basketball analytics but also serves as a practical demonstration of data analysis techniques using Python.

## INSTALLATION

To set up this project on your local machine, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/nba_project.git
   cd nba_project

2. **Install Required Packages**:
Ensure you have Python installed. Create a virtual environment and install the required packages using the requirements.txt file:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
3. **Database Setup**:
Ensure that you have the SQLite database containing the NBA data. Update the db_path variable in both Python files to point to your local database location.

## USAGE
Running the Scripts
1. **Basic Analysis**:
To perform basic analyses on player attributes and game statistics, run:
    ```bash
    python Project_NBA_basic.py
This script performs the following analyses:
- Counts total players who played at least one game.
- Displays the ratio of player positions
- Visualizes the distribution of player heights and weights.
- Lists players who averaged over 20 points with significant rebounds or assists.
- Identifies players with at least 10 seasons of experience.
- Analyzes the correlation between height and rebounds.
- Determines the highest winning teams from 2000 to 2020.

2. **Home Court Advantage Analysis**:
For a more advanced analysis focusing on home and away team performance, run:
    ```bash
    python Project_NBA_HCA.py
This script includes:
- Boxplots comparing points scored by home and away teams.
- Histograms showing win-loss distributions.
- Scatterplots of home vs. away team points.
- User input for filtering data by year and month to analyze win/loss ratios and average points.
- A decision tree model to predict home game outcomes based on points scored.

## DATA VISUALIZATION
The project incorporates various data visualizations to enhance understanding:
- Bar Charts: Display the distribution of player positions.
- Heatmaps: Illustrate the frequency of player positions.
- Histograms: Show the distribution of player heights and weights.
- Boxplots: Compare points scored by home and away teams.
- Scatterplots: Visualize correlations, such as between height and rebounds.

## Example Visualizations
- Distribution of Player Heights:
![historgram_height](https://github.com/user-attachments/assets/4f520686-11e3-42ba-a5ed-d3711289a02c)

- Home vs. Away Points:
![scatterplot_points](https://github.com/user-attachments/assets/65199f50-baf4-4225-8c4f-f986ddc6f0e3)


These visualizations provide a clear, graphical representation of the data, making complex statistics more accessible to users.

## CONCLUSION
The NBA Project serves as a valuable resource for basketball enthusiasts, analysts, and data scientists interested in sports analytics. By combining data extraction, visualization, and predictive modeling, this project not only enhances understanding of player and team performance but also provides a foundation for further exploration in sports analytics.
