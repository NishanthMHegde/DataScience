import numpy as np
import matplotlib.pyplot as plt 
from scipy import stats

salaries = np.random.normal(720000, 20000, 500)
print("Mean of the salaries is %s"%(np.mean(salaries)))
print("Median of the salaries is %s"%(np.median(salaries)))
plt.hist(salaries, 25)
plt.show()
print("Adding a billionaire to the list")
salaries = np.append(salaries, [1000000000])
print("Mean of the salaries is %s"%(np.mean(salaries)))
print("Median of the salaries is %s"%(np.median(salaries)))
number_of_kids = np.random.randint(1,4,50)
print("Mode of the number of kids is " + str(stats.mode(number_of_kids)))

