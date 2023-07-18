# Audrey Kim
# Q1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

pd.set_option('display.max_columns', None)

# 1
data = pd.read_csv('wineQualityReds.csv')
wineDf = pd.DataFrame(data)
# print(wineDf)

# 2
print('Dimensions of wineDf:', wineDf.shape)
# 1599 rows and 13 columns

# 3 - checking for any missing values
print('Any null values:', wineDf.isnull().values.any())
# returns False so no values are null

# 4
X = wineDf.drop(['quality'], axis=1)
X = X.drop(['Wine'], axis=1)
y = wineDf['quality']

# 5
from sklearn.preprocessing import StandardScaler
myScalar = StandardScaler()
myScalar.fit(X)
myScalar.transform(X)
XScaled = pd.DataFrame(myScalar.transform(X), columns=X.columns)
print(XScaled)

# 6
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2022, stratify=y)
X_trainA, X_trainB, y_trainA, y_trainB = train_test_split(X_train, y_train, test_size=0.33,
                                                              random_state=2022, stratify=y_train)

# 7
print('# cases in train partition:', y_train.count())

# 8
print('# cases in test partition:', y_test.count())

# 9
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

trainA_Accuracy = []  # list to store individual accuracy on different value of k
trainB_Accuracy = []
neighbors = range(1, 31)

for k in neighbors:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_trainA, y_trainA)
    y_predA = model.predict(X_trainA)
    y_predB = model.predict(X_trainB)
    # 10
    trainA_Accuracy.append(metrics.accuracy_score(y_trainA, y_predA))
    trainB_Accuracy.append(metrics.accuracy_score(y_trainB, y_predB))

# 11
# plotting the accuracy of trainA and trainB
plt.plot(neighbors, trainA_Accuracy, label='TrainA')
plt.plot(neighbors, trainB_Accuracy, label='TrainB')
plt.xlabel('Number of neighbors (k)')
plt.ylabel('Accuracy')
plt.title('Accuracy of Train A and Train B Data')
plt.legend()
plt.xticks(neighbors)
plt.show()
# based on plot, best k is around 17

# 12 (k = 17)
model = KNeighborsClassifier(n_neighbors=17)
model.fit(X_trainA, y_trainA)
y_pred_test = model.predict(X_test)
print("Accuracy of model:", metrics.accuracy_score(y_test, y_pred_test))

# 13
metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)
plt.show()

# 14
y_pred_A = model.predict(X_trainA)
print("Accuracy of model (train A partition):", metrics.accuracy_score(y_trainA, y_pred_A))

# 15
y_pred_B = model.predict(X_trainB)
print("Accuracy of model (train B partition):", metrics.accuracy_score(y_trainB, y_pred_B))

# 16
y_pred_sample = model.predict([[8, 0.6, 0, 2.0, 0.067, 10, 30, 0.9978, 3.20, 0.5, 10.0]])
print('Quality prediction of sample wine:', y_pred_sample)









