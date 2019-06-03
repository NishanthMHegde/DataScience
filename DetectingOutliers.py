"""
Outliers are data that are distinctly different from the majority of the data. They are usually 2 or more 
standard deviations away from the median/mean value. It is benficial to use the median to detect outliers
because the mean will take into account the outliers when totalling and averaging whereas the median will
not shift much because it involves sorting and finding the middle value.

We have to be careful while considering whether to ignore or keep the outlier value in our dataset.
A typical example would be the situation where we are analysing a dataset of incomes.
The mean of the incomes in this case cannot provide a good picture because a billionaire
can be present which can skew the entire dataset and make the real data insignificant.

We need to decide whether to keep the outliers in our dataset or not depending on the problem at hand.

"""

import numpy as np 
import matplotlib.pyplot as plt 

incomes = np.random.normal(27000, 15000, 10000)
#Let us plot a histogram. We can see that it gives accurate representation
print("Let us plot a histogram. We can see that it gives accurate representation")
plt.hist(incomes, 25)
plt.show()

#Let us now add a billionaire to our data
incomes = np.append(incomes, [100000000])
#We can see that our billionaire has made all other data insignificant
print("We can see that our billionaire has made all other data insignificant")
plt.hist(incomes, 25)
plt.show()

#Let us now detect and remove the outliers
print("Let us now detect and remove the outliers")

median = np.median(incomes)
stddev = np.std(incomes)

#Let us ignore outlier data which are more than 2 standard deviations away from the median.
#It is more helpful to use median in this case
print("Filtering out the outliers and re-constructing our histogram")
filteredData = [data for data in incomes if median - (2*stddev) < data < median + (2*stddev)]
plt.hist(filteredData, 25)
plt.show()
