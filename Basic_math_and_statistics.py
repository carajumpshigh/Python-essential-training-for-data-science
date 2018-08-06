#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Basic math and statistics

import numpy as np
from numpy.random import randn
import pandas as pd
from pandas import Series, DataFrame

import matplotlib.pyplot as plt
import seaborn as sb
from pylab import rcParams


np.set_printoptions(precision=2)

import scipy
from scipy import stats
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr

import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import scale

#%matplotlib.inline
rcParams['figure.figsize'] = 8,4
plt.style.use('seaborn-whitegrid')

# Numpy arithmetic

a = np.array([1,2,3,4,5,6])
b = np.array([[10,20,30],[40,50,60]])
np.random.seed(25)
c = 36 * np.random.randn(6)
# randn-standardized normal distribution(can be negative), rand-[0,1]
d = np.arange(1,35) #[1,35)

print(a * 10)
print(c + a)
print(c * a)    #multiple elements on the same position

aa = np.array([[2.,4.,6.],[1.,3.,5.],[10.,20.,30.]])
bb = np.array([[0.,1.,2.],[3.,4.,5.],[6.,7.,8.]])
print(aa*bb)    #multiple elements on the same position
print(np.dot(aa,bb)) #matix multiple


#=============================================================================


# Summary statistics, descriptive statistics

url = 'http://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/mtcars.csv'
cars = pd.read_csv(url)
cars.columns = ['car_names','mpg','cyl','disp','hp','drat','wt','qsec','vs','am','gear','carb']
cars.head()

print(cars.sum()) # sum of each column
print(cars.sum(axis=1)) # sum of each row
print(cars.median())
print(cars.mean())
print(cars.max())

mpg = cars.mpg
print(mpg.idxmax()) # index of the max element

print(cars.std())
print(cars.var())

gear = cars.gear
print(gear.value_counts()) # return unique items and their counts

print(cars.describe())


#=============================================================================


# Categorical data

cars.index = cars.car_names
cars.head(15)
print(cars)

carb = cars.carb
print(carb.value_counts())

cars_cat = cars[['cyl','vs','am','gear','carb']]
cars_cat.head()
print(cars_cat)

gears_group = cars_cat.groupby('gear')
print(gears_group.describe())

cars['group'] = pd.Series(cars.gear,dtype='category')
print(cars['group'].dtypes)
print(cars['group'].value_counts())

pd.crosstab(cars['am'],cars['gear'])


#=============================================================================


# Parametric methods for correlations 

# Pearson(normally distributed, continuous, numeric variables, linearly related)[1/-1,linear;0,not linear]

sb.pairplot(cars)
x = cars[['mpg','hp','qsec','wt']]
sb.pairplot(x)

mpg = cars['mpg']
hp = cars['hp']
qsec = cars['qsec']
wt = cars['wt']

pearsonr_coefficient, p_value = pearsonr(mpg,hp)
print('PearsonR Correlation Coefficient %0.3f' % (pearsonr_coefficient))

corr = x.corr() #with (),sign the return to corr; without (), sign the function to it
print(corr)

sb.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values)


#=============================================================================


# Nonparametric methods for correlation

# Spearman(ordinal,related nonlinearly,non-normally distributed)[1/-1,related;0,not related]
# Chi-Square tables test for independence(categorical/numeric,bin the numeric variables)[p<0.05,correlated;p>0.05,independent]

x = cars[['cyl','vs','am','gear']]
sb.pairplot(x)

cyl = cars['cyl']
vs = cars['vs']
am = cars['am']
gear = cars['gear']
spearmanr_coefficient, p_value = spearmanr(cyl,vs)
print('Spearman Rank Correlation Coefficient %0.3f' % (spearmanr_coefficient))

table = pd.crosstab(cyl,am)

from scipy.stats import chi2_contingency
chi2,p,dof,expected = chi2_contingency(table.values)
print('Chi-square Statistics %0.3f p_value %0.3f' % (chi2,p))


#=============================================================================


# Scale and transform variables

# Scale - normalization(*1/sum)/standardization(rescale to zero mean and unit variance)
# Scikit-learn preprocessing - scale, center, normalize, bin, impute data

plt.plot(mpg)
print(cars[['mpg']].describe())

mpg_matrix = mpg.values.reshape(-1,1)
scaled = preprocessing.MinMaxScaler()
scaled_mpg = scaled.fit_transform(mpg_matrix)
plt.plot(scaled_mpg)

mpg_matrix = mpg.values.reshape(-1,1)
scaled = preprocessing.MinMaxScaler(feature_range=(0,10))
scaled_mpg = scaled.fit_transform(mpg_matrix)
plt.plot(scaled_mpg)

standardized_mpg = scale(mpg, axis=0, with_mean=False, with_std=False)
plt.plot(standardized_mpg)

standardized_mpg = scale(mpg)
plt.plot(standardized_mpg)













