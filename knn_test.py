import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#ff0000','#00ff00','#0000ff'])

from sklearn import datasets
from sklearn.model_selection import train_test_split


iris = datasets.load_iris()
X,y = iris.data, iris.target

"""
X_train:training samples
X_test: test samples
y_train: training lables
y_test : test lables
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# shape of input--rows & columns - samples & lables
print(X_train.shape)
print(X_train[0])#features of first row

# shape of output-only one colum
print(y_train.shape)
print(y_train)

# shape of input--rows & columns - samples & lables
print(X_train.shape)
print(X_train[0])#features of first row

# shape of output-only one colum
print(y_train.shape)
print(y_train)

# importing KNN module

from knn import KNN

# classifier clf
clf = KNN(k=3)

# fit method training data
clf.fit(X_train,y_train)

#  predict test sample
predictions = clf.predict(X_test)

# test accuracy
accuracy = np.sum(predictions == y_test)/len(y_test)
print(f"my KNN model accuracy is :  {accuracy:.3}")

#  sklearn KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

model_sklearn = KNeighborsClassifier(n_neighbors=3)
model_sklearn.fit(X, y)
predict = model_sklearn.predict(X_test)

print(f"Scikit learn KNN classifier accuracy: {accuracy_score(y_test,predict):.3}")