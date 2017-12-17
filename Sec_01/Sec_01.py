# Section 1R
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
import pandas as pd
# create a simple dataset of people
data = {'Name': ["John", "Anna", "Peter", "Linda"],
        'Location' : ["New York", "Paris", "Berlin", "London"],
        'Age' : [24, 13, 53, 33]
       }

data_pandas = pd.DataFrame(data)
# IPython.display allows "pretty printing" of dataframes
# in the Jupyter notebook
display(data_pandas)
display(data_pandas[data_pandas.Age >30])
#%% Section 1.7
# Load Data
import mglearn
from IPython.display import display
from sklearn.datasets import load_iris
iris_dataset = load_iris()
# Data type View
print("Keys of iris_dataset: {}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193] + "\n...")  # DESCR is a key value for data descriptor
print("Target names: {}".format(iris_dataset['target_names']))
print("Feature names: {}".format(iris_dataset['feature_names']))
print("Type of data: {}".format(type(iris_dataset['data'])))
print("Shape of data: {}".format(iris_dataset['data'].shape))
print("First five rows of data:\n{}".format(iris_dataset['data'][:5]))
#%% Sample set split 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset['data'], iris_dataset['target'], random_state=0)
#%% Data View
# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset["feature_names"])
# create a scatter matrix from the dataframe, color by y_train
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
                           hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
#%% KNN models
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
# prediction
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))
prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(
       iris_dataset['target_names'][prediction]))
# Test set 
y_pred = knn.predict(X_test)
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
#%%







