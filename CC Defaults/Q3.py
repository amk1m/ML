# Audrey Kim
# Q3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

pd.set_option('display.max_columns', None)

# 1
data = pd.read_csv('ccDefaults.csv')
ccDefaults = pd.DataFrame(data)

# 2
print('Any null values:', ccDefaults.isnull().values.any())
# no null values
print("# of non-null samples:", ccDefaults.count())
# The types of the feature data are integers

# 3
print(ccDefaults.head())

# 4
print('Dimensions:', ccDefaults.shape)
# 30000 rows, 25 columns

# 5
ccDefaults = ccDefaults.drop(['ID'], axis=1)

# 6
ccDefaults.drop_duplicates(keep='first', inplace=True)
print('New dimensions:', ccDefaults.shape)
# yes, there were some duplicate records because there are now less rows

# 7
corrMatrix = ccDefaults.corr()
print(corrMatrix)

# 8
# target is dpnm
# top 4 correlated variables with dpnm: PAY_1, PAY_2, PAY_3, PAY_4
X = ccDefaults[['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4']]
y = ccDefaults['dpnm']

# 9
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2022, stratify=y)

# 10
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=2022)
model.fit(X_train, y_train)
y_predTest = model.predict(X_test)

# 11
from sklearn import metrics
print('Accuracy on test partition:', metrics.accuracy_score(y_test, y_predTest))

# 12
confMat = metrics.confusion_matrix(y_test, y_predTest)
print(confMat)
metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_predTest)
plt.show()

# 13
from sklearn.tree import plot_tree
plot_tree(model, feature_names=X.columns, class_names=str(y.unique()), filled=True)
plt.show()



