"""
Used to prevent over-fitting of data. Here a single train test split is not used.
The data is divided into K segments of smalle data randomly.
The K-1 segments are used for training and the remaining segment is used for testing. 
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn import cross_validation
from sklearn import svm 

#Let us try the K-fold cross validaiton using sci-kit learn on Iris dataset using SVM classifier.

#Load the iris data
iris = load_iris()
#Create a single train and test split
X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

#Create an SVM classifier model using linear kernel.
clf = svm.SVC(kernel="linear", C=1).fit(X_train, y_train)
#Check for the classifer's r_squared score
print("Score of Linear classifier on single train test split using linear kernel is %s"%(clf.score(X_test, y_test)))

#We now use the K-fold cross validation on the same SVM classifier model that we created
#In this example we use 5-fold cross validation
scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5)

#check the r-squared scores
print("R-square scores were %s"%(scores))
print("Mean of R-square scores of 5-fold classifier using linear kernel was %s"%(np.mean(scores)))
#We observe that 5 fold cross validation yielded a better mean r-square error.

#Now let us perform a similar activity with a polynomial kernel and check our results

#Create an SVM classifier model using polynomial kernel.
clf = svm.SVC(kernel="poly", C=1).fit(X_train, y_train)
#Check for the classifer's r_squared score
print("Score of Linear classifier on single train test split using polynomial kernel is %s"%(clf.score(X_test, y_test)))

#We now use the K-fold cross validation on the same SVM classifier model that we created
#In this example we use 5-fold cross validation
scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5)

#check the r-squared scores
print("R-square scores were %s"%(scores))
print("Mean of R-square scores of 5-fold classifier using polynomial kernel was %s"%(np.mean(scores)))
#We observe that 5 fold cross validation using an SVM classifier of polynomial kernel yielded a
#worse mean r-square error compared to linear model. This is because of overfitting of data.
print("R-sqaure error of 5 fold validation using linear kernel was higher when compared to polynomial kernel")
print("This is because of overfitting of data.")
