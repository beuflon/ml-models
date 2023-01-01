import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn import linear_model

# Read dataset
df = pd.read_csv("/Users/vunguyen0103/Downloads/honeyproduction.csv")

# Plot based on dataset
prod_per_year = df.groupby('year').totalprod.mean().reset_index()
X = prod_per_year['year']
X = X.values.reshape(-1,1)
y = prod_per_year['totalprod']
plt.scatter(X,y)
# Linear regression model
regr = linear_model.LinearRegression()
regr.fit(X,y)

y_predict = regr.predict(X)
plt.plot(X,y_predict)

# Predict the production in 2050
X_future = np.array(range(2013,2050)) #print the array (or horizontal matrix) 
X_future = X_future.reshape(-1,1) # Print the transpose matrix
future_predict = regr.predict(X_future)
plt.plot(X_future, future_predict)

plt.show()
