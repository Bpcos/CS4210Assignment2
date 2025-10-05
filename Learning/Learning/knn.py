#-------------------------------------------------------------------------
# AUTHOR: Blake Costello
# FILENAME: knn.py
# SPECIFICATION: Using KNN to classify emails into spam or not spam
# FOR: CS 4210- Assignment #2
# TIME SPENT: started 4:05 AM, finished 4:57 AM
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
import pandas as pd

from Learning import X_train

#Reading the data in a csv file using pandas
db = []
df = pd.read_csv('email_classification.csv')
for _, row in df.iterrows():
    db.append(row.tolist())

false_predictions = 0
total_predictions = len(db)
#Loop your data to allow each instance to be your test set
for i, test_instance in enumerate(db):

    training_db = db[:i] + db[i+1:]
    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    X_train = [instance[:-1] for instance in training_db]
    X_train = [[float(val) for val in row] for row in X_train]

    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    y_train = []
    for instance in training_db:
        if instance[-1] == 'spam':
            y_train.append(1)
        else:
            y_train.append(0)

    #Store the test sample of this iteration in the vector testSample
    testSample = [float(val) for val in test_instance[:-1]]
    true_label_str = test_instance[-1]
    true_label_num = 1 if true_label_str == 'spam' else 0

    #Fitting the knn to the data using k = 1 and Euclidean distance (L2 norm)
    clf = KNeighborsClassifier(1)
    clf.fit(X_train, y_train)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    class_predicted = clf.predict([testSample])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    if class_predicted != true_label_num:
        false_predictions += 1

#Print the error rate
error_rate = false_predictions / total_predictions
#print(f'Error rate: {error_rate}')






