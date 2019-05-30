 
import pandas as pd 

#The name of columns that will be used in ratings table
r_cols = ['user_id', 'movie_id', 'ratings']
#create the ratings table
ratings = pd.read_csv('D:\\DataSciencePractice\\DataScience-Python3\\ml-100k/u.data', sep = "\t", names=r_cols, usecols=range(3), encoding="ISO-8859-1")
#The name of columns that will be used in movies table
m_cols = ['movie_id', 'title']
#create the movies table
movies = pd.read_csv('D:\\DataSciencePractice\\DataScience-Python3\\ml-100k/u.item', sep = "|", names=m_cols, usecols=range(2), encoding="ISO-8859-1")
#merge the movies table and ratings table
ratings = pd.merge(movies, ratings)
# print(movieRatings.head())

#Pivot the table so that the index is user_id and the columns are each movie names and column values are ratings given by
#user for each movie
userRatings = ratings.pivot_table(index=['user_id'], columns=['title'], values='ratings')
# print(userRatings.head(10))

#construct co-relation matrix on the user ratings table using peason method and minimum periods as fesible
#co-relation matrix finds co-relation between every pairs of movies
corrMatrix = userRatings.corr(method = 'pearson', min_periods=100)
# print(corrMatrix.head())

#artbitrarily choose a user. Use loc method to find a user from the user_id row
myRatings = userRatings.loc[0].dropna()
# print(myRatings)

#create a new series to hold similar movies
similarCandidates = pd.Series()
#loop through all movies that the selected user has rated
for i in range(0, len(myRatings.index)):
	print("Adding simialrities for " , (myRatings.index[i]))
	#take all columns for the ith movie which user has rated and drop NA values
	similarMovies = corrMatrix[myRatings.index[i]].dropna()
	# scale the value up in relation to the ith movie that user has rated
	print("Scaling with ", myRatings[i])
	similarMovies = similarMovies.map(lambda x: x * myRatings[i])
	#append the similar movies into the similar candidates series
	similarCandidates = similarCandidates.append(similarMovies)
# print(similarCandidates.head())

#Since a movie can appear multiple times, we sum the values by grouping with movie title
similarCandidates = similarCandidates.groupby(similarCandidates.index).sum()
#We sort the recommendation in descending order
similarCandidates.sort_values(inplace=True, ascending=False)
#we drop the movies the user has already rated because we cannot recommend already watched/rated movies.
filteredSimilarities = similarCandidates.drop(myRatings.index)
print(filteredSimilarities.head(10))
