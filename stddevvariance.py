import numpy as np
import matplotlib.pyplot as plt 

marks_scored = np.random.normal(100.0, 20.0, 2000)
plt.hist(marks_scored, 50)
plt.show()
print("Standard deviation is %s"%(marks_scored.std()))
print("Variance is %s"%(marks_scored.var()))