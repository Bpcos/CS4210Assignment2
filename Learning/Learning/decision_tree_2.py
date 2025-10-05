#-------------------------------------------------------------------------
# AUTHOR: blake costello
# FILENAME: decision_tree_2.py
# SPECIFICATION: Using a decision tree to classify contact lens types
# FOR: CS 4210- Assignment #2
# TIME SPENT: started 10:53 pm 
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn import tree
import pandas as pd
import numpy as np


dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

#Reading the test data in a csv file using pandas
dbTest = []
df_test = pd.read_csv('contact_lens_test.csv')
for _, row in df_test.iterrows():
    dbTest.append(row.tolist())

age_map = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3}
prescription_map = {'Myope': 1, 'Hypermetrope': 2}
astigmatism_map = {'No': 1, 'Yes': 2}
tear_map = {'Reduced': 1, 'Normal': 2}
lenses_map = {'No': 1, 'Yes': 2}

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #Reading the training data in a csv file using pandas
    dbTraining = []
    dfTraining = pd.read_csv(ds)
    for _, row in dfTraining.iterrows():
        dbTraining.append(row.tolist())

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    for row in dbTraining:
        X.append([
            age_map[row[0]],
            prescription_map[row[1]],
            astigmatism_map[row[2]],
            tear_map[row[3]]
            ])

    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    for row in dbTraining:
        Y.append(lenses_map[row[4]])

    #Loop your training and test tasks 10 times here
    accuracies = []
    for i in range (10):

       # fitting the decision tree to the data using entropy as your impurity measure and maximum depth = 5
       clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
       clf = clf.fit(X, Y)

       #Read the test data and add this data to dbTest
       #done elsewhere
       correct_predictions = 0
       for data in dbTest:
           #Transform the features of the test instances to numbers following the same strategy done during training,
           #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
           #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           test_instance = [
               age_map[data[0]],
               prescription_map[data[1]],
               astigmatism_map[data[2]],
               tear_map[data[3]]
               ]
           class_predicted = clf.predict([test_instance])[0]

           #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
           if class_predicted == lenses_map[data[4]]:
               correct_predictions += 1

    #Find the average of this model during the 10 runs (training and test set)
    run_accuracy = correct_predictions / len(dbTest)
    accuracies.append(run_accuracy)

    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    avg_accuracy = np.mean(accuracies)
    print(f"final accuracy when training on {ds}: {avg_accuracy}")




