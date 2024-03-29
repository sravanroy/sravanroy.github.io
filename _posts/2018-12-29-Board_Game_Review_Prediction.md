---
title: "Board Game Review Prediction"
date: 2018-12-29
tags: [machine learning, linear regressor, random forest regressor]
header:
  image: "/images/board_game/game.jpg"
excerpt: "Machine Learning, Pandas, Linear and Non-linear regressors"
mathjax: "true"
---

## *This is a simple board game review predictor using machine learning models !*
----
* A [dataset](https://github.com/sravanroy/sravanroy.github.io/tree/master/datasets/board_game_predictor/games.csv) of over 80,000 games is used to train both a linear regressor and a random forest regressor
+ Reviews for a game are then predicted based on significant features in the dataset   
* The models were implemented in Python Jupyter notebook

The required packages are imported into the notebook as shown-
```python
  import sys
  import pandas
  import matplotlib
  import seaborn
  import sklearn 
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  from sklearn.model_selection import train_test_split
```
Now the games [dataset](https://github.com/sravanroy/sravanroy.github.io/tree/master/datasets/board_game_predictor/games.csv) is loaded into the environment
```python
  games = pd.read_csv("games.csv")
  print(games.shape)
```
 
(81312, 20) 

A histogram of all the ratings in the **average_rating** column is plotted, which is the dependent target variable that needs to be predicted
```python
 plt.hist(games["average_rating"])
 plt.show() 
```
![alt]({{ site.url }}{{ site.baseurl }}/images/board_game/hist1.png)


Since most of the ratings are *zero*, the game parameters with rating greater than zero are compared against those with zero rating
```python
  #print(games[games["average_rating"]==0].iloc[0])
  #print(games[games["average_rating"]>0].iloc[0])
```

As most of the data doesn't have any ratings by users, the data is cleaned to remove all the rows without any **users_rated** and also dropped the missing values
```python
games = games[games["users_rated"]>0]
games = games.dropna(axis=0)

plt.hist(games["average_rating"])
plt.show()
```
![alt]({{ site.url }}{{ site.baseurl }}/images/board_game/hist2.png)

Correlation between the game parameters are used for training the models on only key parameters to prevent overfitting
```python
corrmat = games.corr()
fig = plt.figure(figsize = (12,9))

sns.heatmap(corrmat, vmax = .8, square = True)
```
![alt]({{ site.url }}{{ site.baseurl }}/images/board_game/heatmap.png)

The data is filtered for only significant columns which are used to train the models and the **average_rating** column is set as target variable
```python
columns = games.columns.tolist()

columns = [c for c in columns if c not in ["bayes_avearge_rating","average_rating","type","name","id"]]

target = "average_rating" 
```
Create the training and test data sets with 80% train data and 20% test data 
```python
from sklearn.model_selection import train_test_split

train = games.sample(frac =  0.8, random_state = 1)

test = games.loc[~games.index.isin(train.index)]
```
Import the linear regression model and fit the model to the training data
```python
# Import linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Initialize the model class
LR = LinearRegression()

# Fit the model on training data 
LR.fit(train[columns],train[target])
```
Generate predictions for the test set and compute the error between test predictors and actual values
```python
# Generate predictions for the test set
predictions = LR.predict(test[columns])

# Compute error between our test predictors and actual values
mean_squared_error(predictions, test[target])

2.078987486472881
```
The linear regressor is not the best fit on this data, given the complexity and number of rows in it.
Now, a non-linear random forest regression model is implemented in the same way to measure the mean square error
```python
# Import the random forest model
from sklearn.ensemble import  RandomForestRegressor

# Initialize the model
RFR = RandomForestRegressor(n_estimators = 100, min_samples_leaf = 10, random_state = 1)

# Fit to the data
RFR.fit(train[columns],train[target])
# make predictions
predictions = RFR.predict(test[columns])

# compute the error between our test predictors and actual values
mean_squared_error(predictions,test[target])

1.3991311439483058
```
The error is much lower when compared to the previous linear model.
In this case, a random forest model is best fitted for making review predictions of the board games

### Wrapping up

+ *The scope extends to using further non-linear regressors which might have a better accuracy of prediction*
+ *This prediction information comes to handy if we wanted to know the kind of games people liked and get more higher ratings !* 
 



 

