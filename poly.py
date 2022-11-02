# This program identifies a relationship, and use Polynomial regression to
# train, predict, and plot the results.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Training set
x_train = [[900], [1357], [1188], [1600], [1710],[1840],[1980],[2230],[2400],[2930]] #Size of the home
y_train = [[700], [1172], [1250], [1493], [1571],[1711],[1804],[1840],[1956],[1954]] #Electricity consumption(KW Hrs per Month)

# Testing set
x_test = [[805], [1360], [1185], [1602], [1711],[1840],[1985],[2232],[2410],[2929]] #Size of the home
y_test = [[700], [1177], [1256], [1484], [1569],[1711],[1804],[1840],[1954],[1953]] #Electricity consumption(KW Hrs per Month)


# Train the Linear Regression model and plot a prediction
regressor = LinearRegression()
regressor.fit(x_train, y_train)
xx = np.linspace(0, 3000, 5000)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy)

# Set the degree of the Polynomial Regression model
quadratic_featurizer = PolynomialFeatures(degree = 2)

# This preprocessor transforms an input data matrix into a new data matrix of a given degree
x_train_quadratic = quadratic_featurizer.fit_transform(x_train)
x_test_quadratic = quadratic_featurizer.transform(x_test)

# Train and test the regressor_quadratic model
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(x_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))


# Plot the graph
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c = 'orange', linestyle = '-')
plt.title('Home sizes regressed on electricity  hourly consumption per month')
plt.xlabel('Home sizes')
plt.ylabel('Electricy consumption in KW Hrs per Month')
plt.axis([0, 3060, 0, 3000])
plt.grid(True)
plt.scatter(x_train, y_train)
plt.show()
print( x_train)
print( x_train_quadratic)
print(x_test)
print (x_test_quadratic) 
