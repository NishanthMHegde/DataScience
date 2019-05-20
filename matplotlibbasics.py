from scipy.stats import norm
import numpy as np 
import matplotlib.pyplot as plt 

#gaussian/normal probability distribution function
x = np.arange(-3.0, 3.0, 0.001)
plt.plot(x, norm.pdf(x))
plt.show()

# multiple graphs with labels, legend, color, axes, grids

axes = plt.axes()
axes.set_xlabel("CGPA")
axes.set_ylabel("Probability")
axes.grid()
axes.set_ylim([0,1])
axes.set_xlim([-5.0,5.0])
axes.set_xticks([-3,-2,-1,0,1,2,3])
axes.set_yticks([0.5,0.75,1])
x = np.arange(-3.0, 3.0, 0.001)
y = np.arange(-3.0, 3.0, 0.001)
plt.plot(x, norm.pdf(x),'g-')
plt.plot(y, norm.pdf(y ,1,0.5), 'r:')
plt.legend(["class1", "class2"], loc =4)
plt.show()

#bar chart

plt.xlabel("Country")
plt.ylabel("Number of students")
countries = ['India', 'China', 'UK', 'Russia', 'USA']
male_values = [10,20,35,55,80]
female_values = [5,15,15,35,50]
plt.bar(range(0,5), male_values, color = 'blue')
plt.bar(range(0,5), female_values, color = 'pink')
plt.legend(['Male', 'Female'], loc =2)
plt.xticks(range(0,5), countries)
plt.show()

#pie chart
countries = ['India', 'China', 'UK', 'Russia', 'USA']
values = [10,20,35,55,80]
explode = [0.5,0,0,0,0]
colors = ['r', 'b', 'g', 'orange', 'violet']
plt.pie(values, colors = colors, labels = countries, explode = explode)
plt.show()

#scatter plot

x = np.random.randn(50)
y = np.random.randn(50)
plt.scatter(x,y)
plt.show()
