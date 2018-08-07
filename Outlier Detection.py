#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Outlier detection

# main type of outliers: point, contextual, collective
  
  
# Tukey boxplots

# Interquartile range(IQR) * 1.5 = whisker, past these? Outliers.
# IQR = Upper quartile(25%) - Lower quartile(75%)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sb

import sklearn
from sklearn.cluster import DBSCAN
from collections import Counter

#%matplotlib inline
rcParams['figure.figsize'] = 5,4

filepath_or_buffer = '/Users/clu128/Desktop/Exercise Files/Ch05/05_01/iris.data.csv'
df = pd.read_csv(filepath_or_buffer,header=None,sep=',')
df.columns = ['Sepal Length','Sepal Width','Petal Length','Petal Width','Species']
x = df.ix[:,0:4].values
y = df.ix[:,4].values

df.boxplot(return_type='dict')
plt.plot()

Sepal_Width = x[:,1]
iris_outliers = (Sepal_Width > 4)
iris_outliers += (Sepal_Width < 2.05)
print(df[iris_outliers])

pd.options.display.float_format = '{:.1f}'.format
x_df = pd.DataFrame(x)
print(x_df.describe())


#=============================================================================


# Multivariate outliers detection

sb.boxplot(x='Species',y='Sepal Length',data=df,palette='hls')
sb.pairplot(df,hue='Species',palette='hls')


#=============================================================================


# DBSCAN(density based)

# <= 5% are outliers
# eps: max distance to cluster
# min_samples: min number of samples to set core point

model = DBSCAN(eps=0.8,min_samples=19).fit(x)
outliers_df = pd.DataFrame(x)
print(Counter(model.labels_))
print(outliers_df[model.labels_ == -1])

fig = plt.figure()
ax = fig.add_axes([.1,.1,1,1])

colors = model.labels_

ax.scatter(x[:,2],x[:,1],c=colors,s=120)
ax.set_xlabel('Petal Length')
ax.set_ylabel('Sepal Width')
plt.title('DBSCAN for Outlier Detection')



















