#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 Machine learning algorithms - deep learning, ensemble, neural networks,
                               regression, rule system, regulization, bayesian,
                               decision tree, dimension reduction(PCA), instance based(k-Nearest Neighbor),
                               clustering(K-means, hierarchical, DBSCAN)
2/3 training set, 1/3 test set
supervised(labeled), unsupervised(unlabeled)
"""

# Dimensionality Reduction

# Factor analysis

# factors(latent variables)
# assumption: metric, continuous/ordinal features, r>0.3 correlation,
#             >100 observation, >5 observation per feature, homogenous sample

import numpy as np
import pandas as pd

import sklearn
from sklearn.decomposition import FactorAnalysis

from sklearn import datasets

iris = datasets.load_iris()
x = iris.data
variable_names = iris.feature_names
print(x[0:10,])

factor = FactorAnalysis().fit(x)
print(pd.DataFrame(factor.components_, columns=variable_names))


#=============================================================================


# Singular Value Decomposition(SVD) for PCA - unsupervised

# A = u * S * v
# A = original matrix
# u = left orthogonal matrix(important, nonredundant info)
# v = right orthogonal matrix(important, nonredundant info)
# S = diagonal matrix(all info about the decomposition processes performed during the compression)

# use cases: fruad detection, spam detection, image recognition, speech recognition, data preprocessing

import matplotlib.pyplot as plt
import pylab as plt
import seaborn as sb
from IPython.display import Image
from IPython.core.display import HTML
from pylab import rcParams

from sklearn import decomposition
from sklearn.decomposition import PCA

#%matplot inline
rcParams['figure.figsize'] = 5,4
sb.set_style('whitegrid')

pca = decomposition.PCA()
iris_pca = pca.fit_transform(x)

print(pca.explained_variance_ratio_) # contribution of each
print(pca.explained_variance_ratio_.sum()) # info contained >70%

comps = pd.DataFrame(pca.components_, columns=variable_names)

sb.heatmap(comps)









