import numpy as np 
import pandas as pd 
from sklearn import tree 
from sklearn.ensemble import RandomForestClassifier

"""
DecisionTree is a supervised learning algorithm which takes in a training data which consists of a set of variables.
We initially choose a set of variables which are determining attributes. We choose one attribute which we need to determine.
DecisionTree uses Entropy system at each stage. The main aim is to reduce the entropy at each stage of classification when using a
variable. This is called ID3 algorithm.
DecisionTree gives a flowchart which shows whcih attribute was considered at each step, gini value is the entropy and \
number of samples that were used for classification.
"""
"""
Random Forest algorithm works by construting n_estimators number of trees and each of the trees vote among themselves to decide 
which tree can be used to better estimate or classify. Since it is a random forest, we can get different outcomes each time
we run the algorithm.
Main advantage of RandomForests algorithm is that it prevents overfitting of our data.
Overfitting might work beautifully on our training data but it con be disastrous when used on the test data.

"""
#First read the csv data file
df = pd.read_csv('DataScience-Python3/PastHires.csv', header = 0)
print(df.head())

#extract the list of features that we need to use
features = list(df.columns[:6])
print(features)

#Since decision trees required numerical values, we need to map our character values to numerical values
d = {'Y': 1, 'N': 0}
df['Interned'] = df['Interned'].map(d)
df['Top-tier school'] = df['Top-tier school'].map(d)
df['Employed?'] = df['Employed?'].map(d)
df['Hired'] = df['Hired'].map(d)

lod = {'BS': 0, 'MS': 1, 'PhD': 2}
df['Level of Education'] = df['Level of Education'].map(lod)

#We now need to choose the dataset we will use for training and the dataset which contains the feature to predict
#We fit the dataset of features with the Hired attribute
X = df[features]
y = df['Hired']
print(df.head())

#We create a Random Forests classifier
classifier = RandomForestClassifier(n_estimators=10)
# We fit the feature attributes with the attribute we need to analyze/predict.
classifier = classifier.fit(X, y)

# We not carry out a set of predictions.
# For prediction, we pass in a list which consists of the values of features that we have traine our classifier on.
print(classifier.predict([[10, 1, 4, 0, 0, 0]]))
print(classifier.predict([[0, 0, 0, 1, 0, 0]]))
print(classifier.predict([[0, 0, 0, 2, 0, 0]]))
print(classifier.predict([[0, 0, 1, 1, 0, 0]]))
print(classifier.predict([[1, 0, 1, 1, 0, 0]]))