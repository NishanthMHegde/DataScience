import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm 

scaler = StandardScaler()

df = pd.read_excel('http://cdn.sundog-soft.com/Udemy/DataScience/cars.xls')

print(df.head())

X = df[['Mileage','Cylinder', 'Doors']]
y = df['Price']

#x is is not in normalized form, so we need to normalize it y using scaler.fit_transform
# this will convert it into a matrix of values ranging from -1 to 1
print(X.head())
X[['Mileage','Cylinder', 'Doors']] = scaler.fit_transform(X[['Mileage','Cylinder', 'Doors']].as_matrix())
print(X.head())

#We now use statsmodel to compute the coefficient for all our variables that we have selected
# our equation is price = B + B1*mileage + B2*cylinder + B3*doors
# We get values of B1, B2 and B3. We observe that value of B2 is higher because our model
#feels that number of cylinders influenced the price of the car more than tthe doors and mileage.
estimation = sm.OLS(y,X).fit()
print(estimation.summary())

#Let us demonstrate the above observation by checking the mean price of cars for different doors
print(y.groupby(df['Doors']).mean() #We see that the price does not vary greatly for different doors.
