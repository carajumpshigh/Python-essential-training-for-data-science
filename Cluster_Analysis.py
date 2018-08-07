#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Cluster analysis

# K-means method

# base: k(the number of clusters), nearest mean values(Euclidian distance)
# usecase: market price and cost modeling, custom segmentation,
#          insurance claim fruad detection, hedge fund classification
# caution: scale the variables, set k based on scatterplot/data table

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sb

import urllib

import sklearn
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale
import sklearn.metrics as sm
from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report

import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

from sklearn.cluster import AgglomerativeClustering

from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

#%matplot inline
plt.figure(figsize=(7,4))

iris = datasets.load_iris()
x = scale(iris.data)
y = pd.DataFrame(iris.target)
variable_names = iris.feature_names

clustering = KMeans(n_clusters=3,random_state=5)
clustering.fit(x)

iris_df = pd.DataFrame(iris.data)
iris_df.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
y.columns = ['Targets']

color_theme = np.array(['darkgray','lightsalmon','powderblue'])

plt.subplot(1,2,1)
plt.scatter(x=iris_df.Petal_Length,y=iris_df.Petal_Width,c=color_theme[iris.target],s=50)
plt.title('Ground Truth Classification')

plt.subplot(1,2,2)
plt.scatter(x=iris_df.Petal_Length,y=iris_df.Petal_Width,c=color_theme[clustering.labels_],s=50)
plt.title('K-Means Classification')

relabel = np.choose(clustering.labels_,[2,0,1]).astype(np.int64)
plt.subplot(1,2,1)
plt.scatter(x=iris_df.Petal_Length,y=iris_df.Petal_Width,c=color_theme[iris.target],s=50)
plt.title('Ground Truth Classification')

plt.subplot(1,2,2)
plt.scatter(x=iris_df.Petal_Length,y=iris_df.Petal_Width,c=color_theme[relabel],s=50)
plt.title('K-Means Classification')

print(classification_report(y,relabel))
# precision(model's relevancy)[+], recall(model's completeness)[+]


#=============================================================================


# Hierarchical clustering

# distance between each & nearest neighbors -> link
# by dendrogram
# use cases: hospital resource management, business process management,
#            customer segmentation, social network analysis

np.set_printoptions(precision=4, suppress=True)
plt.figure(figsize=(10,3))
#%matplotlib inline
plt.style.use('seaborn-whitegrid')

address = '/Users/clu128/Desktop/Exercise Files/Ch06/06_02/mtcars.csv'
cars = pd.read_csv(address)
cars.columns = ['car_names','mpg','cyl','disp','hp','drat','wt','qsec','vs','am','gear','carb']

x = cars.ix[:,(1,3,4,6)].values
y = cars.ix[:,(9)].values

z = linkage(x, 'ward')
dendrogram(z, truncate_mode='lastp',p=12,leaf_rotation=45.,leaf_font_size=15,show_contracted=True)

plt.title('Truncated Hierarchical Clustering Dendrogram')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')

plt.axhline(y=500)
plt.axhline(y=150)
plt.show()

k=2
Hclustering = AgglomerativeClustering(n_clusters=k,affinity='euclidean',linkage='ward')
Hclustering.fit(x)
print(sm.accuracy_score(y,Hclustering.labels_))

Hclustering = AgglomerativeClustering(n_clusters=k,affinity='euclidean',linkage='complete')
Hclustering.fit(x)
print(sm.accuracy_score(y,Hclustering.labels_))

Hclustering = AgglomerativeClustering(n_clusters=k,affinity='euclidean',linkage='average')
Hclustering.fit(x)
print(sm.accuracy_score(y,Hclustering.labels_))

Hclustering = AgglomerativeClustering(n_clusters=k,affinity='manhattan',linkage='average')
Hclustering.fit(x)
print(sm.accuracy_score(y,Hclustering.labels_))


#=============================================================================


# k-Nearest Neighbor Classification

# use case: stock price prediction, recommendation systems,
#           credit risk analysis, predictive trip planning

# assumption: little noise, labeled, only contain relevant features, has distinguishable subgroups
# caution: avoid large dataset(long time)

x_prime = cars.ix[:,(1,3,4,6)].values
y = cars.ix[:,9].values
x = preprocessing.scale(x_prime)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.33, random_state=17)

clf = neighbors.KNeighborsClassifier()

clf.fit(x_train, y_train)

y_expect = y_test
y_pred = clf.predict(x_test)

print(sm.classification_report(y_expect,y_pred))

