import numpy as np 
import scipy.stats as sp 
import matplotlib.pyplot as plt 

#percentiles
vals = np.random.normal(0, 0.5, 10000)
plt.hist(vals, 50)
print(np.percentile(vals, 50))
print(np.percentile(vals, 90))
print(np.percentile(vals, 1))

#moments : There are 4 types of moments, namely: Mean, Variance, Skew, Kurtosis

#mean
print("mean is %s"%np.mean(vals))

#variance
print("Variance is %s"%vals.var())

#Skew : Negative skew as a longer left tail and positive skew has a longer right tail
print("Skew is %s"%sp.skew(vals))

#Kurtosis: Indicates how sharp the peak is and how thick the tail is.
print("Kurtosis is %s"%sp.kurtosis(vals))

plt.show()