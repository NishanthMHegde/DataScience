import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import scale 
from sklearn.cluster import KMeans

#Kmeans is an unsupervised learning algorithm where we do not feed any learnign data. Clusters are created. 
#Each cluster has a set of points who are somewhat related to each other but we cannot label them. We should decide manually
#First we randomly choose centroids for each cluster and place data points which are closer to a centroid in same cluster
# Then we re-calculate the centroid based on the data points present in the cluster
# We continue this until the number of poisiton interchange is negligible. 

def createClusteredData(N, k):
	# N is total number of poeple and k is number of clusters that is required
	np.random.seed(10)
	X = []
	#number of points in cluster is number of people/number of clusters
	pointsPerCluster = float(N)/k
	for i in range(k):
		#initial centroid is chosen for each cluster. In this example we see how income varies with age and group similar earing people
		incomeCentroid = np.random.uniform(20000, 200000)
		ageCentroid = np.random.uniform(20,60)
		# We now create random datapoints in each cluster. Each cluster will have pointsPerCluster number of data points
		for j in range(int(pointsPerCluster)):
			# we create data poitns which are somewhat close to each other with some standard deviation
			X.append([np.random.normal(incomeCentroid, 10000.0), np.random.normal(ageCentroid, 2.0)])
	# we return the numpy array of the list
	return np.array(X)

data = createClusteredData(100, 5)
# create a KMeans model with n_clusters =5
model = KMeans(n_clusters = 5)
#scale the data to make the data points value uniform so we can work with it.
model = model.fit(scale(data))
# we can see the label assigned to our scaled data
print(model.labels_)
print(len(model.labels_))

#create a scatter plot with first column being income in numpy array and secod column being age in numpy array
# data[:,0] is all values in first column and data[:,1] is all values in second column
#color of point is based on float value of label.
plt.scatter(data[:,0], data[:,1], c = model.labels_.astype(np.float))
plt.show()