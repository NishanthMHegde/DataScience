import numpy as np 
from scipy import stats
import matplotlib.pyplot as plt 

"""
In linear regression, we plot a scatter graph using the two vriables (for example: height and weigh)
We get different points and we try to fit a straight line by taking in to account maximum number of points.
In the future, if we add a new point, then we can predict the weight using the height or the height using
the weight by extending the line. This is called Linear regression. Main aim is to minimse the square of
distance between the points and line. For this we use r_value. We have to take into account maximum number
of variances. This is called Maximum Likelihood estimation.

"""
pageLoadTime = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = 100 - pageLoadTime*3
slope, intercept, r_value, p_value, stderr = stats.linregress(pageLoadTime, purchaseAmount)

print("R-squared is %s"%(r_value**2))
print("Now let us plot our line using slope and intercept value")

def predicted_y(x):
	predicted_y = slope*x + intercept
	return predicted_y
plt.scatter(pageLoadTime, purchaseAmount)
plt.plot(pageLoadTime, predicted_y(pageLoadTime), 'r-')
plt.show()
