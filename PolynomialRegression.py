import numpy as np 
from scipy import stats
import matplotlib.pyplot as plt 

""""Polynimial regression is helpful if we do not have a straight line which fits our curve, but a curvy line
THe order of polynomial determines the complexity of the curve. Polynomial of degree 1 is nothing but Linear
Regression. Higher the order, more complex is the curve. In our urge to fit and include all the outliers,
we cannot increase the degree of polynomial as it will reslt in OVER FITTING of data and for future data, we may
not have accurate estimations.
"""
np.random.seed(2)
pageLoadTime = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = np.random.normal(50.0,10.0,1000)/pageLoadTime
plt.scatter(pageLoadTime, purchaseAmount)
plt.show()

print("Polynimial of degree 4")
x = np.array(pageLoadTime)
y = np.array(purchaseAmount)
p4 = np.poly1d(np.polyfit(x,y,4))

xp = np.linspace(0,7,100)
plt.scatter(x, y)
plt.plot(xp, p4(xp), 'r-')
plt.show()

print("Polynimial of degree 1 (Linear Regression)")
x = np.array(pageLoadTime)
y = np.array(purchaseAmount)
p4 = np.poly1d(np.polyfit(x,y,1))

xp = np.linspace(0,7,100)
plt.scatter(x, y)
plt.plot(xp, p4(xp), 'r-')
plt.show()

print("Polynimial of degree 10 (Over fitting)")
x = np.array(pageLoadTime)
y = np.array(purchaseAmount)
p4 = np.poly1d(np.polyfit(x,y,10))

xp = np.linspace(0,7,100)
plt.scatter(x, y)
plt.plot(xp, p4(xp), 'r-')
plt.show()