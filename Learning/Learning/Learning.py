from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from numpy import mean
from numpy import absolute
from numpy import sqrt
import pandas as pd

df = pd.DataFrame({'x': [0, 0, 1, 2, 2, 3, 3, 4, 4, 4],
                   'y': [5, 3, 4, 4, 1, 3, 2, 4, 3, 1],
                   'class': [0, 0, 0, 1, 0, 1, 1, 1, 1, 0]})

#define predictor and response variables
X = df[['x', 'y']]
y = df['class']

#define cross-validation method to use
cv = LeaveOneOut()

#build multiple linear regression model
model = KNeighborsClassifier(9)

misclassified = []

for train_index, test_index in cv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    true_class = y_test.values[0]
    if prediction[0] != true_class:
        misclassified.append({'index': 1 + test_index[0],
                              'data': X_test.to_dict('records')[0],
                              'predicted_class': prediction[0],
                              'true_class': true_class
        })
error_rate = len(misclassified) / len(df)
#print(f"Error rate: {error_rate}")
#for point in misclassified:
#   print(f"Index: {point['index']}, Data: {point['data']}, Predicted: {point['predicted_class']}, True Class: {point['true_class']}")
