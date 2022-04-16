#!/usr/bin/env python

"""Kernel SVM Classifier
	
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



__author__ = "Anton Skurdal"
__copyright__ = "Copyright 2022, The FAA Project"
__credits__ = ["Anton Skurdal"]
__license__ = "GPL"
__version__ = "1.5"
__maintainer__ = "Anton Skurdal"
__email__ = "antonskurdal@gmail.com"
__status__ = "Development"
#######################
# IMPORTING LIBRARIES #
#######################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the library with the iris dataset
from sklearn.datasets import load_iris


#########################
# IMPORTING THE DATASET #
#########################
# Set up directory
parent_directory = Path("D:/#FAA UAS Project/OpenSky WEEK/Individual Aircraft/batch_1/output")
file = parent_directory / "#AGG.csv"
df = pd.read_csv(file)

""" # Drop erroneous
df = df[df['taxonomy'] != 'erroneous'] """

""" # Drop noise
df = df[df['taxonomy'] != 'noise'] """

print("Taxonomy Counts:\n{}\n".format(df['taxonomy'].value_counts()))

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

######################################################################################
# 								TRAINING THE ALGORITHM								 #
######################################################################################

##########################################
# 			1. LINEAR KERNEL			 #
##########################################

start_time = time.time()
print("Base Model Starting...")
print("Time: {}".format(time.ctime()))
from sklearn.svm import SVC
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.svm import LinearSVC
#svm = SVC(kernel='linear')
svm = LinearSVC(
	random_state=42, 
	C = 75, 
	tol = 1e-2, 
	max_iter = 1000, 
	dual = True,
	class_weight="balanced",
	)
svm.fit(X_train, y_train)

print("--- %s seconds ---" % (time.time() - start_time))

""" ######################
# MAKING PREDICTIONS #
######################

#y_pred = svclassifier.predict(X_test)

y_pred = clf.predict(X_test)

accuracy = clf.score(X_test, y_test) * 100 """



############################
# EVALUATING THE ALGORITHM #
############################

""" from sklearn.metrics import classification_report, confusion_matrix
print("\n########################################\n#\t\t\t1. LINEAR KERNEL\t\t\t#\n########################################")
print("Confusion Matrix:\n{}".format(confusion_matrix(y_test, y_pred, labels = clf.classes_)))
print("Classification Report:\n{}".format(classification_report(y_test, y_pred)))

from sklearn.metrics import ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred, labels = clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = clf.classes_)
disp.plot()
disp.ax_.set_title("Linear Kernel SVM Classifier\nAccuracy: {:.4f}%".format(accuracy))
plt.show() """



# Predict test set
y_pred = svm.predict(X_test)
y_true = y_test

# Show classification report
from sklearn.metrics import classification_report
report = classification_report(y_true, y_pred, labels = svm.classes_)
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
disp.ax_.set_title("Linear SVM Classifier\nAccuracy: {:.4f}%".format(accuracy*100))
plt.show()