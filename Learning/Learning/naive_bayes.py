#-------------------------------------------------------------------------
# AUTHOR: Blake Costello
# FILENAME: naive_bayes.py
# SPECIFICATION: bayesian classification for weather conditions
# FOR: CS 4210- Assignment #2
# TIME SPENT: started 10:20 pm, finished 10:45 pm
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.naive_bayes import CategoricalNB
import pandas as pd


dbTraining = []
dbTest = []

#Reading the training data using Pandas
df = pd.read_csv('weather_training.csv')
for _, row in df.iterrows():
    dbTraining.append(row.tolist())

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
feature_maps = {'Outlook': {'Sunny': 1, 'Overcast': 2, 'Rain': 3},
                'Temperature': {'Hot': 1, 'Mild': 2, 'Cool': 3},
                'Humidity': {'High': 1, 'Normal': 2},
                'Wind': {'Weak': 1, 'Strong':2} }
target_map = {'Yes': 1, 'No': 2}

target_reverse_map = {v: k for k, v in target_map.items()}

X = []
for row in dbTraining:
    transformed_row = [
        feature_maps['Outlook'][row[1]],
        feature_maps['Temperature'][row[2]],
        feature_maps['Humidity'][row[3]],
        feature_maps['Wind'][row[4]]
        ]
    X.append(transformed_row)

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
Y = []
for row in dbTraining:
    Y.append(target_map[row[5]])

#Fitting the naive bayes to the data using smoothing
#--> add your Python code here
clf = CategoricalNB()
clf.fit(X, Y)

#Reading the test data using Pandas
df = pd.read_csv('weather_test.csv')
for _, row in df.iterrows():
    dbTest.append(row.tolist())

#Printing the header for the solution
print(f"{'Day':<11}{'Outlook':<12}{'Temperature':<14}{'Humidity':<11}{'Wind':<10}{'PlayTennis':<13}{'Confidence'}")
print("-" * 80)

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
for row in dbTest:
    test_instance = [
        feature_maps['Outlook'][row[1]],
        feature_maps['Temperature'][row[2]],
        feature_maps['Humidity'][row[3]],
        feature_maps['Wind'][row[4]]
        ]
    predicted_prob = clf.predict_proba([test_instance])[0]
    confidence = max(predicted_prob)

    if confidence >= 0.75:
        prediction_num = clf.predict([test_instance])[0]
        prediction_str = target_reverse_map[prediction_num]
        print(f"{row[0]:<11}{row[1]:<12}{row[2]:<14}{row[3]:<11}{row[4]:<10}{prediction_str:<13}{confidence:.2f}")

