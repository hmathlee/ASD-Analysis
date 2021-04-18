## Import necessary packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

## pcaCoord(pca, b) returns the coordinates of vector b with respect to the orthonormal basis vectors for Principal
##   Component Analysis pca

def pcaCoord(pca, b):
    '''
    Expresses a vector, b, in terms of the component vectors obtained Principal Component Analysis (PCA) on a dataset (i.e. gives coordinates)
    
    pcaCoord(pca, b) takes in a PCA, pca, and a vector, b (in the form of a numpy array), and computes the coordinates of b with respect to the set of pca component vectors
    Returns the coordinate vector of b with respect to set of pca component vectors
    
    Parameters:
      pca: PCA
      b: numpy array
    
    Returns:
    numpy array
      coordinate vector of b with respect to pca basis
    
    '''
    
    c = np.zeros(b.size)
    j = 0
    for p in pca.components_:
        c[j] = np.dot(b, p)
        j = j + 1
    return c

## Read in the dataset

df = pd.read_csv(r'PATH\TO\ASD\DATA\CSV\FILE)

## Encode for any features taking non-numeric values ##

for feature in df.columns:
    if isinstance(df[feature][0], str) and feature != 'Class/ASD':
        df[feature] = LabelEncoder().fit_transform(df[feature])
W = df.loc[:, df.columns != 'age_desc']
X = W.loc[:, W.columns != 'Class/ASD']
X = StandardScaler().fit_transform(X)
X = pd.DataFrame(X)
y = df['Class/ASD']
df = pd.concat([X, y], axis=1)

## K-Fold Cross Validation against our classification model ##

scores = []
model = LogisticRegression()
pca = PCA(n_components=3)
cv = KFold(n_splits=10, random_state=42, shuffle=True)
for train_index, test_index in cv.split(X):
    X_train, X_test, y_train, y_test = X.values[train_index], X.values[test_index], y[train_index], y[test_index]
    X_train = pca.fit_transform(X_train)

    X_test = np.array(X_test)
    for i in range(0, X_test.shape[0]):
        X_test[i] = pcaCoord(pca, X_test[i])
    testCoord = (pd.DataFrame(X_test)).iloc[:, 0:pca.n_components_]

    model.fit(X_train, y_train)
    scores.append(model.score(testCoord, y_test))

## Print out the metrics from k-fold cross-validation ##

print("\n=============== CROSS-VALIDATION METRICS ===============")
print("Min Score: ", min(scores))
print("Max Score: ", max(scores))
print("Mean Score: ", np.mean(scores))
print("==========================================================")

## ---------- END OF CLASSIFIER SCRIPT ---------- ##

## The following is some old code that I originally had. It plots the data after applying PCA. ##

X = pca.fit_transform(X)
X = pd.DataFrame(data=X, columns=['pc1', 'pc2', 'pc3'])
df = pd.concat([X, y], axis=1)

## Set up the plot ##

fig = plt.figure(figsize=(8, 8))
ax = Axes3D(fig)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_zlabel('Principal Component 3', fontsize=15)
ax.set_title('Three-Component PCA for ASD Data', fontsize=20)

## Plot post-PCA data points ##

targets = ['YES', "NO"]
colors = ['r', 'b']
for target, color in zip(targets, colors):
    indicesToKeep = df['Class/ASD'] == target
    ax.scatter(df.loc[indicesToKeep, 'pc1'], df.loc[indicesToKeep, 'pc2'], df.loc[indicesToKeep, 'pc3'], c=color, s=50)
ax.legend(targets)
ax.grid()
plt.show()
