import pandas as pd
import numpy as np
data = pd.read_csv("/home/salarydata.csv")
X = data['YearsExperience'].values.reshape(-1,1)
y = data['Salary']
from sklearn.linear_model import LinearRegression
brain = LinearRegression()
brain.fit(X,y)
x = brain.predict([[float(input("Enter Years of Experience :- "))]])
print(x[0])
