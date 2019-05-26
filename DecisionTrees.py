import numpy as np 
import pandas as pd 
from sklearn import tree 
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

"""
DecisionTree is a supervised learning algorithm which takes in a training data which consists of a set of variables.
We initially choose a set of variables which are determining attributes. We choose one attribute which we need to determine.
DecisionTree uses Entropy system at each stage. The main aim is to reduce the entropy at each stage of classification when using a
variable.
DecisionTree gives a flowchart which shows whcih attribute was considered at each step, gini value is the entropy and \
number of samples that were used for classification.
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

#We create a decision Tree classifier
classifier = tree.DecisionTreeClassifier()
# We fit the feature attributes with the attribute we need to analyze/predict.
classifier = classifier.fit(X, y)

#This is the boiler plate code we use to display the Decision Tree Graph.
dot_data = StringIO()
tree.export_graphviz(classifier, out_file=dot_data,  
                feature_names = features)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
graph.write_png("hiring.png")