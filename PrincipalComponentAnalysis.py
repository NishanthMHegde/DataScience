"""
Principal Component analysis is one of the methods to reduce the number of dimensions (or columns)
present in the dataset and at the same time preserve the variance in the dataset.
So we can reduce a higher dimensional dataset to a lower dimension for data analysis.
The lower dimension data can then be used to reconstruct the higher dimension data with very little
losses because most of the variance would have bee preserved.
"""

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from itertools import cycle
import pylab as pl 

#load the iris dataset which contains information about
#the length and width of petals and sepals. (4 dimensional data)

iris = load_iris()
numSamples , numFeatures = iris.data.shape
print("Number of samples are %s"%(numSamples))
print("Number of dimensions in data are %s"%(numFeatures))
print("The different target names or flowers are %s"%(iris.target_names))
#We now PCA to reduce the number of dimensions from 4 to 2
X= iris.data
pca = PCA(n_components=2, whiten=True).fit(X)
X_pca = pca.transform(X)
print("Iris targets are %s"%(iris.target))
print(X_pca)
print(X_pca[iris.target==0, 0])
#We can now see that we have a 2D data which was obtained by dimensionality reduction of
#4D data
print("The different PCA components now are %s"%(pca.components_))
#We can see that around 92% of variance along with an additional 5% of
#variance was preserved. Totally 97% of variance was preserved by PCA.
print("The variations preserved are %s"%(pca.explained_variance_ratio_))
percentage_variance_preserved = sum(pca.explained_variance_ratio_)*100
print("The total percentage of variations preserved is %s %%"%(percentage_variance_preserved))

# now let us plot our new 2D data on a scatter plot and check if it matches previous data
#our legend will contain the class of the 3 different flowers each coded in one of rgb
#X_pca contains data for each iris target
target_ids = range(len(iris.target_names))
colors = cycle('rgb')
pl.figure()

for i, c, label in zip(target_ids,colors, iris.target_names):
	pl.scatter(X_pca[iris.target==i, 0], X_pca[iris.target==i, 1], c=c, label=label)
pl.legend()
pl.show()