# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 17:18:45 2019

@author: Muhammad Ali Haider
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('aids.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'cyan')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('(Training set) Deaths due to Aids per year')
plt.xlabel('Year')
plt.ylabel('Deaths')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'magenta')
plt.plot(X_train, regressor.predict(X_train), color = 'black')
plt.title('(Test set) Deaths due to Aids per year')
plt.xlabel('Year')
plt.ylabel('Deaths')
plt.show()