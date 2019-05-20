import numpy as np 
import matplotlib.pyplot as plt 

#Covariance defines how to variables are related to each other and whether the 2 variables are tighlty bound and dependent \
#or if they are loosely related to each other. High covariance means the 2 variables are not so much correlated.

#our own covariance function

def d_mean(x):
	xmean = np.mean(x)
	d_mean = [xi-xmean for xi in x]
	return d_mean
def covariance(x,y):
	n = len(x)
	return np.dot(d_mean(x), d_mean(y))/(n-1)

# we can analyse the pageloadtime and purchaseamount to see the covariance
print("No co-relation")
pageLoadTime = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = np.random.normal(100,10,1000)
print("Our calculated covariance is %s"%(covariance(pageLoadTime, purchaseAmount)))
print("Numpy calculated covariance is %s"%(np.cov(pageLoadTime, purchaseAmount)))
plt.scatter(pageLoadTime, purchaseAmount)
plt.show()

#Let us try to make them somewhat related
print("Somewhat co-related")
print("Our calculated covariance is %s"%(covariance(pageLoadTime, purchaseAmount)))
print("Numpy calculated covariance is %s"%(np.cov(pageLoadTime, purchaseAmount)))
purchaseAmount = np.random.normal(100,10,1000)/pageLoadTime
plt.scatter(pageLoadTime, purchaseAmount)
plt.show()

#Covariance does not give a good idea about how much 2 variables differ or are co-related. \
#We need to use co-relation co-efficient.
#Co-relation coefficient is 1: Perfectly corelated
#Co-relation coefficient is 0: Not co-related
#Co-relation coefficient is -1: Perfectly inversely corelated

#Let us try to make them linearly co-related

def correlation(x,y):
	stdx = x.std()
	stdy = y.std()
	return covariance(x,y)/stdx/stdy
print("linearly co-related")
purchaseAmount = 100 - pageLoadTime*3
#we can see that the pageload time and purchase amount are perfectly inversely co-related
print("Our calculated co-relation  is %s"%(correlation(pageLoadTime, purchaseAmount)))
print("Numpy calculated co-relation is %s"%(np.corrcoef(pageLoadTime, purchaseAmount)))
plt.scatter(pageLoadTime, purchaseAmount)
plt.show()

