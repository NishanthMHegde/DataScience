import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score

pageLoadTime = np.random.normal(3.0, 1.0, 10000)
purchaseAmount = np.random.normal(50.0, 10.0, 10000)/pageLoadTime

X = np.array(pageLoadTime)
Y = np.array(purchaseAmount)

#divide the data into 80% training data and 20% test data
print("Fitting the training data. (Supervised learning)")
trainX = X[:8000]
trainY = Y[:8000]
p8 = np.poly1d(np.polyfit(trainX,trainY,8))
xp = np.linspace(0, 7, 10000)
plt.scatter(trainX,trainY)
plt.plot(xp, p8(xp), 'r-')
plt.show()

print("Using the coefficients obtained by polynomial regression on the trained data to fit the test data")

testX = X[8000:]
testY = Y[8000:]
plt.scatter(testX,testY)
plt.plot(xp, p8(xp), 'r-')
plt.show()

print("checking r2 score (r squared error) on the test data")

r2 = r2_score(testY, p8(testX))
print(r2)
print("checking r2 score (r squared error) on the training data")

r2 = r2_score(trainY, p8(trainX))
print(r2)