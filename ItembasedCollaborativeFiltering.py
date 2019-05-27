import numpy as np 
import pandas as pd 
"""
Item based collaborative filtering: This method of recommendation system is used to recommend one product \
which is similar to a product which is being watched or used by a user.

It is useful because:
a. Number of items is less than number of people.
b. Items do not change over time unlike the mood of people.
c. Very hard to game/fool the recommender system.
"""

# ratings table column names
r_cols = ['user_id', 'movie_id', 'rating']
#create a pandas dataframe for ratings. The data is seperated by a tab space and encoded.
ratings = pd.read_csv('D:\\DataSciencePractice\\DataScience-Python3\\ml-100k/u.data', sep = "\t", \
	names=r_cols, usecols = range(3), encoding='latin-1')
# movie table column names
m_cols = ['movie_id', 'title']
#create a pandas dataframe for movies. The data is seperated by a PIPE and encoded.
movies = pd.read_csv('D:\\DataSciencePractice\\DataScience-Python3\\ml-100k/u.item', sep = "|", names=m_cols, \
	usecols = range(2), encoding='latin-1')
print(ratings.head(5))
print(movies.head(5))

#We not merge both the tables. The merge will happen on movie_id
ratings = pd.merge(ratings, movies)
print(ratings.head(10))

#In this step, we change the orientation of the table. We make the index as user_id. The table 
#has all the movie titles as the columns. The value of the columns is the rating which the user has 
#given to each movie.
movieRatings = ratings.pivot_table(index= 'user_id' , columns='title', values='rating')
print(movieRatings.head(10))
# print(movieRatings.columns)

#We not pick a movie and check rating given by each user
starWarsMovie = movieRatings['Star Wars (1977)']
print(starWarsMovie.head(5))

#We now pick our movie and check how much the rating is correlated with the rating the user 
#has given to other movies (other columns)
similarMovies = movieRatings.corrwith(starWarsMovie)
#We drop the NaN values which can arise due to missing ratings of a user.
similarMovies = similarMovies.dropna()

print(similarMovies.head())

#We can see that if we sort the correlation in descending order, many movies
# have a correlation of 1 even though they arent so related.
print(similarMovies.sort_values(ascending=False))

#We create a movie stats table for each movie title and on the rating column, we find the rating size and mean rating
#This will allow us to filter movie titles which have been rated at least a minimum threshold amount of times.
movieStats = ratings.groupby('title').agg({'rating': [np.size, np.mean]})
print(movieStats.head(5))

#We filter movies which have been rated at least 100 times.
popularMovies = movieStats['rating']['size']>=100
#We can sort the popular movies based on the mean of the rating.
print(movieStats[popularMovies].sort_values([('rating', 'mean')], ascending=False))

#We now create a new dataframe by joining the popular movies with the correlation table
#for the movie we chose . All  movies present in movieStats are joined with values in similarMovies table.
df = movieStats[popularMovies].join(pd.DataFrame(similarMovies, columns = ['similarity']))
print(df.head(10))

#We can now sort the values and see which movie is more closely related to a given movie.
print(df.sort_values(['similarity'], ascending = False))