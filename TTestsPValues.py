import numpy as np 
from scipy import stats 

"""
T-Tests give an idea on how much better or how much worse our new tests can be.
A higher T-value says that our new experiment has more impact and a less 
T-value has less impact on our tests. 
We perform T-tests so we can ascertain how much impact our change has on our business.

P-values on the other hand tell us the probability that these values that we retrieve from
our t-tests are not due to random variations. A lower p-value says that our results are less
prone to random variation and can be considered for results.

Here A is our control group (older website)
B is the test/experiment group.

FIX a threshold for P-value. A value which falls below the p-value threshold means we can consider the
results.
"""

#Let us have A and B as order amounts where B is slightly higher
A = np.random.normal(25.0, 5.0, 1000)
B = np.random.normal(26.0, 5.0, 1000)

#We now get high negative value of T statistics and lower p-value which tell that change is bad with less random variation.
print(stats.ttest_ind(A, B))

#Now let us make B almost same as A.
B = np.random.normal(25.0, 5.0, 1000)
print(stats.ttest_ind(A, B))
#We see a little higher p-value and less t-stats which tell us that we cannot really consider this test.

print("We can now see that increasing the sample size will also not help our cause. In such cases, after running \n the experiment"
		" for a set period of time, we stop the tests.")


A = np.random.normal(25.0, 5.0, 10000)
B = np.random.normal(25.0, 5.0, 10000)
print(stats.ttest_ind(A, B))

A = np.random.normal(25.0, 5.0, 100000)
B = np.random.normal(25.0, 5.0, 100000)
print(stats.ttest_ind(A, B))