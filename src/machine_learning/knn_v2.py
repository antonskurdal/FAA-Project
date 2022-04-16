#!/usr/bin/env python

"""K Nearest Neighbor Clustering
    
    Original Code Source:
	https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/
    
    description
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn import metrics
from sklearn import neighbors


__author__ = "Anton Skurdal"
__copyright__ = "Copyright 2022, The FAA Project"
__credits__ = ["Anton Skurdal"]
__license__ = "GPL"
__version__ = "1.5"
__maintainer__ = "Anton Skurdal"
__email__ = "antonskurdal@gmail.com"
__status__ = "Development"


# Original Code Source
# https://chrisalbon.com/code/machine_learning/nearest_neighbors/k-nearest_neighbors_classifer/

#################
# PRELIMINARIES #
#################

import pandas as pd
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn



##################
# CREATE DATASET #
##################

# Set up directory
parent_directory = Path("D:/#FAA UAS Project/OpenSky WEEK/Individual Aircraft/batch_1/output")
file = parent_directory / "#AGG.csv"
df = pd.read_csv(file)

""" # Drop erroneous
df = df[df['taxonomy'] != 'erroneous'] """

""" # Drop noise
df = df[df['taxonomy'] != 'noise'] """

print("Taxonomy Counts:\n{}\n".format(df['taxonomy'].value_counts()))


#################
# PLOT THE DATA #
#################

#seaborn.lmplot('test_1', 'test_2', data=training_data, fit_reg=False,hue="outcome", scatter_kws={"marker": "D","s": 100})
""" seaborn.lmplot(x='test_1', y='test_2', data=training_data, fit_reg=False,hue="outcome", scatter_kws={"marker": "D","s": 100})
plt.show() """
# hue_order = ['normal', 'erroneous', 'noise', 'dropout']
# seaborn.lmplot(x='time', y='dropout_length', data=train, fit_reg=False,hue="taxonomy", scatter_kws={"marker": "D","s": 100}, hue_order=hue_order)
# plt.show()

####################
# TRAIN TEST SPLIT #
####################
# Define base dataset and features
features = ['lat', 'lon', 'geoaltitude', 'velocity', 'dropout_length']
X = df[features]
y = df['taxonomy']

# Train/Test Split
# X_train -> training features
# y_train -> training labels
# X_test -> testing features
# y_test -> testing labels
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("\n")
_df = X_train
print("X_train Set Size: {:.4f}% ({}/{})".format(((_df.shape[0]/df.shape[0]) * 100), _df.shape[0], _df.shape[0]))
_df = y_train
print("y_train Taxonomy Counts:\n{}\n".format(_df.value_counts()))
_df = X_test
print("X_test Set Size: {:.4f}% ({}/{})".format(((_df.shape[0]/df.shape[0]) * 100), _df.shape[0], _df.shape[0]))
_df = y_test
print("y_test Taxonomy Counts:\n{}\n".format(_df.value_counts()))



from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
X_train = scaling.transform(X_train)
X_test = scaling.transform(X_test)





###############
# Loop Models #
###############

from sklearnex import patch_sklearn
patch_sklearn()
from sklearn import neighbors

start_time = time.time()
print("Loop Model Starting...")

# Ranges for K and record testing accuracy
k_range = range(1, 100)

# We can create Python dictionary using [] or dict()
scores = []

# We use a loop through the range 1 to 26
# We append the scores in the dictionary
for k in k_range:
    start_time = time.time()
    knn = neighbors.KNeighborsClassifier(n_neighbors=k, n_jobs=16, p = 2, weights='distance', leaf_size=50)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
    print("Model k={} finished in {:.4f} seconds".format(k, (time.time() - start_time)))
#print("--- %s seconds ---" % (time.time() - start_time))


# import Matplotlib (scientific plotting library)
import matplotlib.pyplot as plt

# plot the relationship between K and testing accuracy
# plt.plot(x_axis, y_axis)
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()



exit()










#################
# Train the KNN #
#################

""" clf = neighbors.KNeighborsClassifier(3, weights = 'uniform')
trained_model = clf.fit(X, y) """
""" n_neighbors = 3
clf = neighbors.KNeighborsClassifier(n_neighbors, weights = 'uniform')
trained_model = clf.fit(X, y)
 """
start_time = time.time()
print("Base Model Starting...")
n_neighbors = 10
knn = neighbors.KNeighborsClassifier(n_neighbors)#, weights = 'uniform')
knn.fit(X_train, y_train)
print("--- %s seconds ---" % (time.time() - start_time))


############################
# EVALUATING THE ALGORITHM #
############################

# Predict test set
y_pred = knn.predict(X_test)
y_true = y_test

# Show classification report
from sklearn.metrics import classification_report
report = classification_report(y_true, y_pred, labels = knn.classes_)
print("Classification Report:\n{}".format(report))

# Show accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy: {}\n".format(accuracy))

# Plot confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#cm = confusion_matrix(y_true, y_pred)#, labels = base_model.classes_)
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
#disp.plot()
disp.ax_.set_title("K Nearest Neighbor Classifier\nAccuracy: {:.4f}%".format(accuracy*100))
plt.show()




exit()








##########################
# VIEW THE MODEL'S SCORE #
##########################

#trained_model.score(X, y)
""" print(trained_model.score(X, y)) """

print(trained_model.score(X, y))
accuracy = trained_model.score(X, y) * 100

#########################################
# APPLY THE LEARNER TO A NEW DATA POINT #
#########################################

""" # Create a new observation with the value of the first independent variable, 'test_1', as .4 
# and the second independent variable, test_1', as .6 
x_test = np.array([[.4,.6]])


# Apply the learner to the new, unclassified observation.
#trained_model.predict(x_test)
print(trained_model.predict(x_test))


#trained_model.predict_proba(x_test)
print(trained_model.predict_proba(x_test)) """

x_test = test[features]
print(trained_model.predict(x_test.values))
print(trained_model.predict_proba(x_test.values))

preds = trained_model.predict(x_test.values)

#codes_test, _y_test = pd.factorize(test['taxonomy'])
#y_test = pd.factorize(test['taxonomy'])[0]
y_test = test['taxonomy']


from sklearn.metrics import classification_report
print("Classification Report:\n{}".format(classification_report(y_test, preds)))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, preds, labels = clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = clf.classes_)
disp.plot()
disp.ax_.set_title("K Nearest Neighbor Classifier (k = {})\nAccuracy: {:.4f}%".format(n_neighbors, accuracy))

plt.show()



hue_order = ['normal', 'erroneous', 'noise', 'dropout']
ax = sns.stripplot(test['taxonomy'], preds, hue_order=hue_order)
ax.set(xlabel ='Actual Taxonomy', ylabel ='Prediction') 
# print(_y.take(codes).unique())
# print(codes)
# print(_y)
# print(y.take(codes))

plt.show()