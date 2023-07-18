# Audrey Kim
# Q2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

pd.set_option('display.max_columns', None)

# 1
data = pd.read_csv('Breast_Cancer.csv')
df = pd.DataFrame(data)
# print(df)

# 2
print('Any null values:', df.isnull().values.any())
# false means no null values

# 3 - target variable is diagnosis
X = df.drop(['diagnosis'], axis=1)
y = df['diagnosis']
print(X)

# 4
sn.countplot(data=df, x='diagnosis')
plt.title('Countplot of Diagnosis')
plt.show()

# from sklearn.preprocessing import StandardScaler
# myScalar = StandardScaler()
# myScalar.fit(X)
# xScaled = pd.DataFrame(myScalar.transform(X), columns=X.columns)

# 5
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2022, stratify=y)

# 6
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=800)
model.fit(X_train, y_train)

# 7
from sklearn import metrics
y_predTrain = model.predict(X_train)
y_predTest = model.predict(X_test)
# printing classification report both for train and then for test sets
print(metrics.classification_report(y_train, y_predTrain))
print(metrics.classification_report(y_test, y_predTest))

# 8
confMat = metrics.confusion_matrix(y_test, y_predTest)
print(confMat)
metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_predTest)
plt.show()


