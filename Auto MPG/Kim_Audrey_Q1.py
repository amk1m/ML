# Audrey Kim
# ITP 449 Fall 2022
# HW7
# Q1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import ResidualsPlot

# reading in the data into a dataframe
data = pd.read_csv('auto-mpg.csv')
df = pd.DataFrame(data)

# summarizing the data
print(df.describe())
# A. B. the mean of mpg is 23.514573 and the median of mpg is 23.000000
# C. the mean of mpg is higher, which means that the plot should be skewed right
# plotting a histogram showing the distribution of mpg values
plt.hist(df['mpg'])
plt.ylabel('Frequency')
plt.xlabel('mpg values')
plt.show()
# the histogram shows that the data is skewed right (meaning that there are relatively
# more mpg values that are on the lower end of the spectrum)


# D. droppinng the non-numeric / irrelevant columns in df
pDF = df.drop(['No'], axis=1)
pDF = pDF.drop(['car_name'], axis=1)
# plotting the pairplot for the dataframe
sn.pairplot(pDF)
plt.show()

# E. Based on the pairplot matrix, the variables that seem the most linearly correlated
# are displacement and weight because the pairplot between these two variables seems
# to be the most linear out of all of the plots in the matrix.

# F. Based on the pairplot matrix, the variables that seem the least linearly correlated
# are model year and weight.

# G. creating a scatterplot with displacement as the x-axis and mpg as the y-axis
xscatter = df['displacement']
yscatter = df['mpg']
plt.scatter(xscatter, yscatter)
plt.xlabel('Displacement')
plt.ylabel('Miles per Gallon')
plt.title('Miles per Gallon vs Displacement')
plt.show()

# H. creating a linear regression model with displacement as the predictor and mpg as the target variable
model = LinearRegression()
x = df['displacement']
X = x[:, np.newaxis]
y = df['mpg']
model.fit(X, y)
Y = model.predict(X)
plt.xlabel('Displacement')
plt.ylabel('Miles per Gallon')
plt.scatter(X, y)
plt.plot(X, Y)
plt.title('Miles per Gallon vs Displacement (including Linear Regression)')
plt.show()

print('Intercept:', model.intercept_)
print('Slope:', model.coef_)
# a.b. For the linear regression equation, the intercept is approx 35.17475 and the coefficient is approx -0.06028.
# c. The regression equation is y = -0.06028x + 35.17475
# d. For the model, as displacement increases, the predicted value for mpg decreases.

# e. For a car with displacement of 200, the mpg is predicted to be 23.11875
# getting the predicted value by the regression at displacement=200
y_pred0 = model.predict([[200]])
print(y_pred0)

# f. (done above)

# g. plotting the residuals
viz = ResidualsPlot(model)
viz.fit(X, y)
viz.show()








