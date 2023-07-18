# Audrey Kim
# ITP 449 Fall 2022
# HW7
# Q2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

pd.set_option('display.max_columns', None)

# 1. reading the data into a dataframe
data = pd.read_csv('Titanic.csv')
df = pd.DataFrame(data)
# print(df)
# 2. 'Survived' is the target variable
# 3. drop the 'Passenger' variable b/c not likely important for regression
df = df.drop(['Passenger'], axis=1)
print(df)

# 4. checking if there are any missing values in df
print(df.isnull().values.any()) # prints false, which means there are no missing values

# 5. count plots for remaining factors (class, sex, age)
# count plot for class
sn.countplot(x=df['Class'])
plt.title('Count Plot: Class')
plt.show()
# count plot for sex
sn.countplot(x=df['Sex'])
plt.title('Count Plot: Sex')
plt.show()
# count plot for age
sn.countplot(x=df['Age'])
plt.title('Count Plot: Age')
plt.show()

# 6. converting the categorical variables to dummy variables
df2 = pd.get_dummies(df, columns=['Class', 'Sex', 'Age'])
print(df2)

# 7. partitioning the data into train and test sets
X = df2.iloc[:, 1:]
y = df2.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2022)

# 8. fitting the data to a logistic regression model
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)
y_pred = LogReg.predict(X_test)

# 9. displaying the classification report, which has the accuracy, precision, and recall
# for predictions of survivability
print(classification_report(y_test, y_pred))

# 10. displaying the confusion matrix
plot_confusion_matrix(LogReg, X_test, y_test)
plt.show()

# 11. displaying the predicted value for the survivability of an adult female passenger traveling 2nd class
# (predicted value is yes)
y_pred0 = LogReg.predict([[0, 1, 0, 0, 1, 0, 1, 0]])
print(y_pred0)









