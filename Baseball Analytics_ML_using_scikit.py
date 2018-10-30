#Part 1: Predicting MLB Team Wins per Season
# import `pandas` and `sqlite3`
import pandas as pd
import sqlite3

# Connecting to SQLite Database
conn = sqlite3.connect('lahman2016.sqlite')
# Querying Database for all seasons where a team played 150 or more games and is still active today.
query = '''select * from Teams 
inner join TeamsFranchises
on Teams.franchID == TeamsFranchises.franchID
where Teams.G >= 150 and TeamsFranchises.active == 'Y';
'''
# Creating dataframe from query.
Teams = conn.execute(query).fetchall()

#Using pandas, you then convert the results to a DataFrame
# and print the first 5 rows using the head() method:
# Convert `Teams` to DataFrame
teams_df = pd.DataFrame(Teams)
# Print out first 5 rows
#print(teams_df.head)

#Cleaning and Preparing The Data

#add headers by passing a list of your headers to the columns attribute from pandas.
# Adding column names to dataframe
cols = ['yearID','lgID','teamID','franchID','divID','Rank','G','Ghome','W','L','DivWin','WCWin','LgWin','WSWin','R','AB','H','2B','3B','HR','BB','SO','SB','CS','HBP','SF','RA','ER','ERA','CG','SHO','SV','IPouts','HA','HRA','BBA','SOA','E','DP','FP','name','park','attendance','BPF','PPF','teamIDBR','teamIDlahman45','teamIDretro','franchID','franchName','active','NAassoc']
teams_df.columns = cols
# Print the first rows of `teams_df`
#print(teams_df.head())
# Print the length of `teams_df` to find out how many rows I'm dealing with
#print(len(teams_df))
#Prior to assessing the data quality, let’s first eliminate the columns
#  that aren’t necessary or are derived from the target column (Wins).
# Dropping your unnecesary column variables.
drop_cols = ['lgID','franchID','divID','Rank','Ghome','L','DivWin','WCWin','LgWin','WSWin','SF','name','park','attendance','BPF','PPF','teamIDBR','teamIDlahman45','teamIDretro','franchID','franchName','active','NAassoc']
df = teams_df.drop(drop_cols, axis=1)
# Print out first rows of `df`
#print(df.head())
#remove null values
# Print out null values of all columns of `df`
#print(df.isnull().sum(axis=0).tolist())
#If you eliminate the rows where the columns have a small number of null values,
# you’re losing a little over five percent of our data.
# Since you’re trying to predict wins,
# runs scored and runs allowed are highly correlated with the target.
#  You want data in those columns to be very accurate.
#Strike outs (SO) and double plays (DP) aren’t as important.
#Better off keeping the rows and filling the null values with the median value from each
# of the colums by using the fillna() mehtod. Caught stealing (CS) and hit by
# pitch (HBP) aren’t very important variables either.
# With so many null values in these columns,
# it’s best to eliminate the columns all together.

# Eliminating columns with null values
df = df.drop(['CS','HBP'], axis=1)
# Filling null values with the '.fillna()' method
df['SO'] = df['SO'].fillna(df['SO'].median())
df['DP'] = df['DP'].fillna(df['DP'].median())
# Print out null values of all columns of `df`
#print(df.isnull().sum(axis=0).tolist())


#Exploring and Visualizing The Data
import matplotlib.pyplot as plt
#You’ll start by plotting a histogram of the target column so you can see the
# distribution of wins.
plt.hist(df['W'])
plt.xlabel('Wins')
plt.title('Distribution of Wins')
#plt.show()
plt.close()
#Print out the average wins (W) per year. You can use the mean() method for this.
#print(df['W'].mean())

#It can be useful to create bins for your target column while exploring your data, but you need to make sure not to
#  include any feature that you generate from your target column when you train the model.
# Including a column of labels generated from the target column in your training set would be like giving your model
# the answers to the test.

#To create your win labels, you’ll create a function called assign_win_bins which will take in an integer value (wins)
# and return an integer of 1-5 depending on the input value.
#Next, you’ll create a new column win_bins by using the apply() method on the wins column and passing in
# the assign_win_bins() function.
# Creating bins for the win column
def assign_win_bins(W):
    if W < 50:
        return 1
    if W >= 50 and W <= 69:
        return 2
    if W >= 70 and W <= 89:
        return 3
    if W >= 90 and W <= 109:
        return 4
    if W >= 110:
        return 5
# Apply `assign_win_bins` to `df['W']`
#print(df['W'])
df['win_bins'] = df['W'].apply(assign_win_bins)
df = df[df['yearID'] > 1900]
#Now let’s make a scatter groaph with the year on the x-axis and wins on the y-axis and highlight
# # # the win_bins column with clors.
plt.scatter(df['yearID'], df['W'], c=df['win_bins'])
plt.title('Wins Scatter Plot')
plt.xlabel('Year')
plt.ylabel('Wins')
#plt.show()
plt.close()

#As you can see in the above scatter plot, there are very few seasons from before 1900 and the game was much different
# back then. Because of that, it makes sense to eliminate those rows from the data set.
# Filter for rows where 'yearID' is greater than 1900

#When dealing with continuous data and creating linear models,
# integer values such as a year can cause issues.
# It is unlikely that the number 1950 will have the same relationship to the
# rest of the data that the model will infer.
#You can avoid these issues by creating new variables that label the data based on the yearID value.
#Let’s make a graph below that indicates how much scoring there was for each year.
#You’ll start by creating dictionaries runs_per_year and games_per_year.
#  Loop through the dataframe using the iterrows() method. Populate the runs_per_
# year dictionary with years as keys and how many runs were scored that year as
#  the value. Populate the games_per_year dictionary with years as keys and how
# many games were played that year as the value.

# Create runs per year and games per year dictionaries
runs_per_year = {}
games_per_year = {}

for i, row in df.iterrows():
    year = row['yearID']
    runs = row['R']
    games = row['G']
    if year in runs_per_year:
        runs_per_year[year] = runs_per_year[year] + runs
        games_per_year[year] = games_per_year[year] + games
    else:
        runs_per_year[year] = runs
        games_per_year[year] = games

#print(runs_per_year)
#print(games_per_year)

#Next, create a dictionary called mlb_runs_per_game.
# Iterate through the games_per_year dictionary with the items() method.
# Populate the mlb_runs_per_game dictionary with years as keys and the number of
# runs scored per game, league wide, as the value.
# Create MLB runs per game (per year) dictionary
mlb_runs_per_game = {}
for k, v in games_per_year.items():
    year = k
    games = v
    runs = runs_per_year[year]
    mlb_runs_per_game[year] = runs / games

#print(mlb_runs_per_game)
#Finally, create your plot from the mlb_runs_per_game dictionary by putting the
#years on the x-axis and runs per game on the y-axis.
# Create lists from mlb_runs_per_game dictionary
lists = sorted(mlb_runs_per_game.items())
x, y = zip(*lists)
# Create line plot of Year vs. MLB runs per Game
plt.plot(x, y)
plt.title('MLB Yearly Runs per Game')
plt.xlabel('Year')
plt.ylabel('MLB Runs per Game')
#plt.show()
plt.close()
#Now that you have a better idea of scoring trends, you can create new variables
#  that indicate a specific era that each row of data falls in based on the yearID.
#  You’ll follow the same process as you did above when you created the win_bins
# column.
#This time however, you will create dummy columns; a new column for each era.
# You can use the get_dummies() method for this.
# Convert years into decade bins and creating dummy variables
# Creating "year_label" column, which will give your algorithm information about
# how certain years are related
# (Dead ball eras, Live ball/Steroid Eras)
def assign_label(year):
    if year < 1920:
        return 1
    elif year >= 1920 and year <= 1941:
        return 2
    elif year >= 1942 and year <= 1945:
        return 3
    elif year >= 1946 and year <= 1962:
        return 4
    elif year >= 1963 and year <= 1976:
        return 5
    elif year >= 1977 and year <= 1992:
        return 6
    elif year >= 1993 and year <= 2009:
        return 7
    elif year >= 2010:
        return 8
# Add `year_label` column to `df`
df['year_label'] = df['yearID'].apply(assign_label)
dummy_df = pd.get_dummies(df['year_label'], prefix='era')
# Concatenate `df` and `dummy_df`
df = pd.concat([df, dummy_df], axis=1)
#print(df.head())
# Create column for MLB runs per game from the mlb_runs_per_game dictionary
def assign_mlb_rpg(year):
    return mlb_runs_per_game[year]
df['mlb_rpg'] = df['yearID'].apply(assign_mlb_rpg)


# Convert years into decade bins and creating dummy variables
def assign_decade(year):
    if year < 1920:
        return 1910
    elif year >= 1920 and year <= 1929:
        return 1920
    elif year >= 1930 and year <= 1939:
        return 1930
    elif year >= 1940 and year <= 1949:
        return 1940
    elif year >= 1950 and year <= 1959:
        return 1950
    elif year >= 1960 and year <= 1969:
        return 1960
    elif year >= 1970 and year <= 1979:
        return 1970
    elif year >= 1980 and year <= 1989:
        return 1980
    elif year >= 1990 and year <= 1999:
        return 1990
    elif year >= 2000 and year <= 2009:
        return 2000
    elif year >= 2010:
        return 2010


df['decade_label'] = df['yearID'].apply(assign_decade)
decade_df = pd.get_dummies(df['decade_label'], prefix='decade')
df = pd.concat([df, decade_df], axis=1)

# Drop unnecessary columns
df = df.drop(['yearID', 'year_label', 'decade_label'], axis=1)
# Create new features for Runs per Game and Runs Allowed per Game
df['R_per_game'] = df['R'] / df['G']
df['RA_per_game'] = df['RA'] / df['G']
# Create scatter plots for runs per game vs. wins and runs allowed per game vs. wins
fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.scatter(df['R_per_game'], df['W'], c='blue')
ax1.set_title('Runs per Game vs. Wins')
ax1.set_ylabel('Wins')
ax1.set_xlabel('Runs per Game')

ax2.scatter(df['RA_per_game'], df['W'], c='red')
ax2.set_title('Runs Allowed per Game vs. Wins')
ax2.set_xlabel('Runs Allowed per Game')
#plt.show()
plt.close()

#Before getting into any machine learning models,
# it can be useful to see how each of the variables is correlated with the
#  target variable. Pandas makes this easy with the corr() method.
#print(df.corr()['W'])

#Add labels derived from a K-means cluster algorithm with sklearn
#First, create a DataFrame that leaves out the target variable (wins)
attributes = ['G','R','AB','H','2B','3B','HR','BB','SO','SB','RA','ER','ERA','CG',
'SHO','SV','IPouts','HA','HRA','BBA','SOA','E','DP','FP','era_1','era_2','era_3','era_4','era_5','era_6','era_7','era_8','decade_1910','decade_1920','decade_1930','decade_1940','decade_1950','decade_1960','decade_1970','decade_1980','decade_1990','decade_2000','decade_2010','R_per_game','RA_per_game','mlb_rpg']
data_attributes = df[attributes]
#print(df.head())

#Determine how many clusters you want.
#You can get a better idea of your ideal # of clusters by using sklearn's
#'ssilhouette_score() function, which returns the mean silhouette coefficient
#over all samples. You want a higher silhouette score, and the score decreases as
#more clusters are added.
# Import necessary modules from `sklearn`
from sklearn.cluster import KMeans
from sklearn import metrics
# Create silhouette score dictionary
s_score_dict = {}
for i in range(2,11):
    km = KMeans(n_clusters=i, random_state=1)
    l = km.fit_predict(data_attributes)
    s_s = metrics.silhouette_score(data_attributes, l)
    s_score_dict[i] = [s_s]
# Print out `s_score_dict`
#print(s_score_dict)

#Initialize the model
#Set number of clusters to 6, and random state to 1
# Create K-means model and determine euclidian distances for each data point
#Determine the Euclidian distances for each data point by using the fit_transform()
# method and then visualize the clusters with a scatter plot.
kmeans_model = KMeans(n_clusters=6, random_state=1)
distances = kmeans_model.fit_transform(data_attributes)
# Create scatter plot using labels from K-means model as color
labels = kmeans_model.labels_
plt.scatter(distances[:,0], distances[:,1], c=labels)
plt.title('Kmeans Clusters')
#plt.show()
plt.close()

#add the labels from your clusters into the data set as a new column
#also add the string 'labels' to the attributes list, for use later
df['labels'] = labels
attributes.append('labels')
#print(df.head())

#separate train and test data

# Create new DataFrame using only variables to be included in models
numeric_cols = ['G', 'R', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB', 'RA', 'ER', 'ERA', 'CG', 'SHO', 'SV', 'IPouts',
                'HA', 'HRA', 'BBA', 'SOA', 'E', 'DP', 'FP', 'era_1', 'era_2', 'era_3', 'era_4', 'era_5', 'era_6',
                'era_7', 'era_8', 'decade_1910', 'decade_1920', 'decade_1930', 'decade_1940', 'decade_1950',
                'decade_1960', 'decade_1970', 'decade_1980', 'decade_1990', 'decade_2000', 'decade_2010', 'R_per_game',
                'RA_per_game', 'mlb_rpg', 'labels', 'W']
data = df[numeric_cols]
#print(data.head())

# Split data DataFrame into train and test sets
#This time, you will simply take a random sample of 75 percent of our data for the train data set and use the other
#  25 percent for your test data set. Create a list numeric_cols with all of the columns you will use in your model.
# Next, create a new DataFrame data from the df DataFrame with the columns in the numeric_cols list.
# Then, also create your train and test data sets by sampling the DataFrame data
train = data.sample(frac=0.75, random_state=1)
test = data.loc[~data.index.isin(train.index)]
x_train = train[attributes]
y_train = train['W']
x_test = test[attributes]
y_test = test['W']

#Selecting Error Metric and Model

#Mean Absolute Error(MAE) is the metric I'll use to determine how accurate my
#model is. MAE measures how close the predictions are to the eventual outcomes.
#Specifically, for this data, that means that this error metric will provide
# you with the average absolute value that your prediction missed its mark.

#First model will be a Linear Regression
# Import `LinearRegression` from `sklearn.linear_model`
from sklearn.linear_model import LinearRegression
# Import `mean_absolute_error` from `sklearn.metrics`
from sklearn.metrics import mean_absolute_error
# Create Linear Regression model, fit model, and make predictions
lr = LinearRegression(normalize=True)
lr.fit(x_train, y_train)
predictions = lr.predict(x_test)
# Determine mean absolute error
mae = mean_absolute_error(y_test, predictions)
# Print `mae`
print('LinearRegression MAE: '+str(mae))

#Next model will be a Ridge Regression
#The RidgeCV model allows you to set the alpha parameter, which is a complexity
# parameter that controls the amount
# of shrinkage (read more here).
# The model will use cross-validation to deterime which of the alpha
# parameters you provide is ideal.
# Import `RidgeCV` from `sklearn.linear_model`
from sklearn.linear_model import RidgeCV
# Create Ridge Linear Regression model, fit model, and make predictions
rrm = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0), normalize=True)
rrm.fit(x_train, y_train)
predictions_rrm = rrm.predict(x_test)
# Determine mean absolute error
mae_rrm = mean_absolute_error(y_test, predictions_rrm)
print('RidgeCV MAE: '+str(mae_rrm))
