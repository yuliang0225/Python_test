# Section 1
"""
Spyder Editor
Smuch!
This is a temporary script file.
"""
# Section 1.4
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

data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data,(row_indices,col_indices)))
print("COO :\n {}".format(eye_coo))
#%% Test Matplotlib 
import matplotlib.pyplot as plt
x = np.linspace(-10,10,100)
y=np.sin(x)
plt.plot(x,y,marker="x",color="black")
#%% Test Pandas



#%% 