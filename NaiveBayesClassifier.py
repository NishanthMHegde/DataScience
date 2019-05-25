import numpy as np
import pandas as pd 
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

def extractMessage(filepath):
	# We walk through all the files present in the filepath
	# os.walk gives the root, directory name and filename
	for root, dirnames, filenames in os.walk(filepath):
		# We examing each of the files
		for filename in filenames:
			#We construct the destination of the file
			path = os.path.join(root, filename)
			messages = []
			inBody = False #Used to skip header information and cut to the chase
			with open(path, 'r', encoding = 'latin1') as read_file:
				lines = read_file.readlines()
				for line in lines:
					
					if inBody:
						messages.append(line)
					elif line == "\n":
						inBody = True
			required_messages = "\n".join(messages)
			# We return the file destination and the containing message for each loop
			# This is a generator function which keeps returning file path and messages for \
			#each filename. Hence we use yield and not return 
			yield path, required_messages

def readFiles(path, msg_class):
	#We need to maintain a list of dictionaries with the keys as message and class.
	#This will be used to populate our training dataset with a set of training values.
	rows = []
	index = []
	#We extract each of the message present in the spam and ham folders.
	# Spam and ham folders have files in them which have respective pam/ham message.
	for filename, message in extractMessage(path):
		rows.append({'message': message, 'class': msg_class})
		index.append(filename)
	#We return a dataframe by using the rows of dictionary and we keep the filename as index.
	return pd.DataFrame(rows, index = index)

#We first construct an empty Pandas dataframe which consists of two rows which is message and the class to which it belongs

data = pd.DataFrame({'message':[], 'class': []})
#We then need to append all the messages present in the spam files to the dataframe and classify it as spam
data = data.append(readFiles("D:\\DataSciencePractice\\DataScience-Python3\\emails\\spam", 'spam'))
#We then need to append all the messages present in the ham files to the dataframe and classify it as ham
data = data.append(readFiles("D:\\DataSciencePractice\\DataScience-Python3\\emails\\ham", 'ham'))

print(data.head(10))

#We first need to convert our messages into numbers and get to know the number of times they occur.

vectorizer = CountVectorizer()
#We keep a count of how many times each word in a message appears by converting words to number format
#CountVectorizer does this counting stuff for us
counts = vectorizer.fit_transform(data['message'].values)

classifier = MultinomialNB()
targets = data['class'] # spam/ham is our target classification based on their frequency of occurence in the respective spam/ham class
classifier.fit(counts, targets)
print(classifier)

# We now use sample messages to train our spam classifier and see how it works.
examples = ['Get Free Viagra ', 'Nishanth, can we play cricket?']
examples_count = vectorizer.transform(examples)
# We use the classifier that we trained using our dataframe to predict spam/ham for new example messages
predictions = classifier.predict(examples_count)
print(predictions)
