#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Basic algorithmic learning

# Linear regression

# assumption: all variables are continuous numeric, not categorical;
#             free of missing values and outliers;
#             linear relationship between predictors and predictant;
#             all predictors are independent for each other;
#             residuals are normally distributed.

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import urllib

import scipy
from scipy.stats import spearmanr

import matplotlib.pylab as plt
from pylab import rcParams
import seaborn as sb

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import scale
from collections import Counter

#%matplotlib inline
rcParams['figure.figsize'] = 5,4
sb.set_style('whitegrid')

address = '/Users/clu128/Desktop/Exercise Files/Ch08/08_01/enrollment_forecast.csv'
enroll = pd.read_csv(address)
enroll.columns = ['year','roll','unem','hgrad','inc']
enroll.head()

sb.pairplot(enroll)    #check linear relationships
print(enroll.corr())   #check correlations

enroll_data = enroll.ix[:,(2,3)].values
enroll_target = enroll.ix[:,1].values
enroll_data_names = ['unem','hgrad']

x, y = scale(enroll_data), enroll_target

missing_values = x==np.NAN #check if any missing data
print(x[(missing_values) == True])

LinReg = LinearRegression(normalize=True)
LinReg.fit(x,y)
print(LinReg.score(x,y)) #performance


#=============================================================================


# Logistic regression

# assumption: free of missing values;
#             the predictant variable is binary or ordinal;
#             all predictors are independent of each other;
#             >= 50 observations per predictor variable(reliable).

address = '/Users/clu128/Desktop/Exercise Files/Ch08/08_02/mtcars.csv'
cars = pd.read_csv(address)
cars.columns = ['car_names','mpg','cyl','disp','hp','drat','wt','qsec','vs','am','gear','carb']
cars.head()

cars_data = cars.ix[:,(5,11)].values
cars_data_names = ['drat','carb']

y = cars.ix[:,9].values

sb.regplot(x='drat',y='carb',data=cars)     #check if ordinal
drat = cars['drat']
carb = cars['carb']
spearmanr_coefficient, p_value = spearmanr(drat,carb)  #check correlationship
print('Spearman Rank Correlation Coefficient %0.3f' % (spearmanr_coefficient))
print(cars.isnull().sum())  #check if any missing values
sb.countplot(x='am',data=cars,palette='hls')   #check if binary
cars.info()    #check if the dataset size is sufficient

x = scale(cars_data)
LogReg = LogisticRegression()
LogReg.fit(x,y)
print(LogReg.score(x,y))

y_pred = LogReg.predict(x)
from sklearn.metrics import classification_report
print(classification_report(y,y_pred))


#=============================================================================


# Naive Bayes classifiers

# type: Multinomial (describe discrete frequency counts)
#       Bernoulli (prediction from binary features)
#       Gaussian (prediction from normally distributed features)

# assumption: predictors are independent of each other;
#             the past conditions still hold true.

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data'
raw_data = urllib.request.urlopen(url)
dataset = np.loadtxt(raw_data,delimiter=',')
print(dataset[0])

x = dataset[:,0:48]
y = dataset[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.33,random_state=17)

BernNB = BernoulliNB(binarize=True)
BernNB.fit(x_train,y_train)
print(BernNB)

y_expect = y_test
y_pred = BernNB.predict(x_test)
print(accuracy_score(y_expect,y_pred))

MultiNB = MultinomialNB()
MultiNB.fit(x_train,y_train)
print(MultiNB)

y_expect = y_test
y_pred = MultiNB.predict(x_test)
print(accuracy_score(y_expect,y_pred))

GausNB = GaussianNB()
GausNB.fit(x_train,y_train)
print(GausNB)

y_expect = y_test
y_pred = GausNB.predict(x_test)
print(accuracy_score(y_expect,y_pred))

BernNB = BernoulliNB(binarize=0.1)
BernNB.fit(x_train,y_train)
print(BernNB)

y_expect = y_test
y_pred = BernNB.predict(x_test)
print(accuracy_score(y_expect,y_pred))

