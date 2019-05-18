import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import norm
from scipy.stats import expon
from scipy.stats import binom
from scipy.stats import poisson
#uniform probability distribution

x = np.random.uniform(-10.0, 10.0, 5000)
plt.hist(x, 50)
plt.show()

#gaussian/normal probability distribution function
x = np.arange(-3.0, 3.0, 0.001)
plt.plot(x, norm.pdf(x))
plt.show()

#binomial probability mass function
x = np.arange(-3.0, 3.0, 0.001)
plt.plot(x, expon.pdf(x))
plt.show()

#binomial probability mass function
x = np.arange(0, 10.0, 0.001)
plt.plot(x, binom.pmf(x, 10, 0.5))
plt.show()

#poisson probability mass function
mean = 400
x = np.arange(300, 500, 0.5)
plt.plot(x, poisson.pmf(x,mean ))
plt.show()