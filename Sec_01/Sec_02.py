#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 20:48:51 2017

@author: smuch
"""
# Section 2 Supervised Learning
#%% Import necessary toolbox
import mglearn
import matplotlib.pyplot as plt
from IPython.display import display
import pandas as pd
import numpy as np
#%% Sec 2.3
# generate dataset
X, y = mglearn.datasets.make_forge()
# plot dataset
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape: {}".format(X.shape))
#%%
X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Feature")
plt.ylabel("Target")
#%% Cancer data
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys(): {}".format(cancer.keys()))
print("Shape of cancer data: {}".format(cancer.data.shape)) # 569*30
print("Sample counts per class:\n{}".format(
      {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))})) #357 : 212
print("Feature names:\n{}".format(cancer.feature_names)) # Features
#%% Boston data
from sklearn.datasets import load_boston
boston = load_boston()
print("Data shape: {}".format(boston.data.shape)) # 506*13
#%% KNN
mglearn.plots.plot_knn_classification(n_neighbors=1)
#%%
mglearn.plots.plot_knn_classification(n_neighbors=3)






