# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%% Test NP
import numpy as np
x = np.array([[1,2,3],[4,5,6]])
print(x)
#%% Test SciPy
from scipy import sparse
eye = np.eye(4)
print ("NumPy array:\n{}".format(eye))

sparse_martix = sparse.csr_matrix(eye)
print("\nSciPy sparse.csr martix:\n{}".format(sparse_martix))

#%% Test Matplotlib 

#%%