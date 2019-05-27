import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets, svm
"""
SVM is a supervised learning algorithm. It supports higher dimensions (more number of attributes based on which
we can classify data ). It is computationally expensive. It has different Kernels like linear, polynomial which
can give different accuracy.

"""
def createDataSet(N,k):
    X = []
    y = []
    pointsPerCluster = float(N)/k
    for i in range(k):
        incomeCentroid = np.random.uniform(20000.0, 2000000.0)
        ageCentroid = np.random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):
            X.append([np.random.normal(incomeCentroid, 150000.0), np.random.normal(ageCentroid, 5.0)])
            y.append(i)

    return np.array(X), np.array(y)

#boiler plate code to plot grids which show SVM classification
def plotPredictions(clf):
    xx, yy = np.meshgrid(np.arange(0, 250000, 10),
                     np.arange(10, 70, 0.5))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    plt.figure(figsize=(8, 6))
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:,0], X[:,1], c=y.astype(np.float))
    plt.show()


(X, y) = createDataSet(100, 5)

plt.scatter(X[:,0], X[:,1], c= y.astype(np.float))
plt.show()

C = 1.0
svm = svm.SVC(kernel = 'linear', C=C).fit(X, y)
plotPredictions(svm)

