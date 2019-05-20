import numpy as np 
from numpy import random

totals = {10:0, 20:0, 30:0, 40:0, 50:0, 60:0}
purchases = {10:0, 20:0, 30:0, 40:0, 50:0, 60:0}
totalPurchases = 0
for i in range(100000):
	ageDecade = random.choice([10,20,30,40,50,60])
	totals[ageDecade] = totals[ageDecade] + 1
	purchaseProbability = float(ageDecade)/100.0
	if purchaseProbability > random.random():
		purchases[ageDecade] = purchases[ageDecade] + 1
		totalPurchases = totalPurchases + 1

print(totals)
print(purchases)
print(totalPurchases)

#find P(E|F) where E = probability of purchasing and F = probability \
#that you are in your 30s

required_probability = float(purchases[30])/float(totals[30])
print("probability of purchasing an item when in 30s is %s"%(required_probability))

print("probability of being in 30s is %s"%(float(totals[30])/100000.0))
print("probability of purchasing something regardless of age is %s"%(float(totalPurchases)/100000.0))