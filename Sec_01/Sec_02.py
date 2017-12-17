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

