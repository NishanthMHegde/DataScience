import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

df = pd.read_csv('DataScience-Python3/PastHires.csv')
print(df['Level of Education'].value_counts())
df['Level of Education'].value_counts().plot(kind = 'hist')
plt.show()