import pandas as pd 
import numpy as np 
import operator
from scipy import spatial 

"""
K Nearest Neighbour is a very simple supervised learning algorithm. It is used to classify an item
based on the K nearest neighbours available to it in the scatter plot.
We first have a classified set of points on the scatter plot, for example, we can have the classification
as movie genre type.
When a new movie is required to be classified, then we can select K neighbours who are near to a point on the scatter plot,
and those neighbours can vote to decide the classification of the new point.
Different forms of voting include selecting the most popular class of the neighbours, 
calculating the average of the neighbours ,etc.

"""

# names of the ratings columns
r_cols = ['user_id', 'movie_id', 'ratings']
# create the ratings table
ratings = pd.read_csv('D:\\DataSciencePractice\\DataScience-Python3\\ml-100k/u.data', sep = "\t",
					  names=r_cols, usecols=range(3))
print(ratings.head())

# create a movieStats table by grouping the movie_id and calculating the mean size
#and mean rating of each movie. This is done by aggregating on the 'ratings' column
movieStats = ratings.groupby('movie_id').agg({'ratings': [np.size, np.mean]})
print(movieStats.head())

#create a dataframe for number of ratings for each movie.
movieNumRatings = pd.DataFrame(movieStats['ratings']['size'])
# movieNormalizedRatings = movieNumRatings.apply(lambda x: print("X is %s"%(np.max(x))))
# Scale the number of times each movie was rated to a value between 0 and 1.
#0 means rated very less, 1 means rated more number of times.
movieNormalizedRatings = movieNumRatings.apply(lambda x: (x- np.min(x))/(np.max(x) - np.min(x)))
print(movieNormalizedRatings.head())

#create a movie dictionary which is empty. The key will be the movie_id
movieDict = dict()
#open the file which contains movieID, name , genres and other info
with open('D:\\DataSciencePractice\\DataScience-Python3\\ml-100k/u.item') as read_file:
	lines = read_file.readlines()
	for line in lines:
		line = line.rstrip('\n')
		#split each line based on | symbol
		fields = line.split("|")
		movieID = int(fields[0])
		name = fields[1]
		genres = fields[5:25]
		# convert each genre field to integer. 0 means does not belong to the genre
		#1 means it belongs to the genre
		genres= [ int(x) for x in genres]
		movieDict[movieID] = [name, genres, movieNormalizedRatings.loc[movieID]['size'], movieStats.loc[movieID]['ratings']['mean']]

print(movieDict[1])

# function to compute distance between two points
def computeDistance(a,b):
	# select the genre from the list item
	genreA = a[1]
	genreB = a[2]
	# you can use cosine distance to calculate distance between the two genres
	genreDistance = spatial.distance.cosine(genreA, genreB)
	# print("Genre cosine distance is %s"%(genreDistance))
	# select the number of ratings from the list item
	popularityA = a[2]
	popularityB = b[2]
	popularityDistance = abs(popularityA - popularityB)
	# print("Popularity distance is %s"%(popularityDistance))
	totalDistance = genreDistance + popularityDistance
	return totalDistance

print(computeDistance(movieDict[2], movieDict[4]))

# function get the K nearest neighbours
def getNearestNeighbours(K, movieID):
	distances = []
	#loop through each of the movie in the movie dictionary we created
	for movie in movieDict.keys():
		#for all movies that are not same to the movie ID we need to classify
		if movieID!=movie:
			#compute the distance between the two movies
			dist = computeDistance(movieDict[movie], movieDict[movieID])
			# add the movie_id, distance to the distances list
			distances.append((movie, dist))
	# sort the distances in ascending order by using the distance we calculated as the key
	distances.sort(key=operator.itemgetter(1))
	neighbours = []
	for i in range(K):
		#append the movie_id for the K nearest neighbours
		neighbours.append(distances[i][0])
		#return list of neighbours which has movieID
	return neighbours

K = 10
avgRating = 0
# find the K nearest neighbours
neighbours = getNearestNeighbours(K, 4)
# loop through each neighbour and calculate the average rating
for neighbour in neighbours:
	avgRating = avgRating + movieDict[neighbour][3]
	print("%s has an average rating of %s"%(movieDict[neighbour][0], movieDict[neighbour][3] ))

avgRating = float(avgRating/K)
#check if the computed average rating is closer to the actual rating
print("Average rating of selected new movie from neighbours is %s"%(avgRating))
print("Average rating of selected new movie from movie Dictionary is %s"%(movieDict[4][3]))
