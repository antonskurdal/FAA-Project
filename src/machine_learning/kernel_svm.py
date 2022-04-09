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

""" #url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
#colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Create an object called iris with the iris data
iris = load_iris()

# Create a dataframe with the four feature variables
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add a new column with the species names, this is what we are going to try to predict
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# View the top 5 rows
print(df.head()) """


# Open directory and concatenate all files into a dataframe
directory = Path.cwd() / "data" / "ML-datasets" / "RandomForest"
files = [f for f in directory.glob('**/*.csv')]
df = pd.concat(map(pd.read_csv, files), ignore_index = True)

# Drop irrelevant data
try:
    df = df.drop('Unnamed: 0', axis = 1)
except KeyError as e:
    print(e)
    pass
df = df.dropna(axis = 0, how = 'any', subset = ['lat', 'lon', 'geoaltitude', 'velocity', 'dropout_length'])

# print(df)
# print(df.shape[0])
print(df['taxonomy'].value_counts())
#data = df.copy(deep=True)

#################
# PREPROCESSING #
#################
""" df['is_train'] = np.random.uniform(0, 1, len(df)) <= 0.7
train, test = df[df['is_train'] == True], df[df['is_train'] == False]

X_train = train
X_test = test

codes_train, _y_train = pd.factorize(train['taxonomy'])
y_train = pd.factorize(train['taxonomy'])[0]

codes_test, _y_test = pd.factorize(test['taxonomy'])
y_test = pd.factorize(test['taxonomy'])[0] """



""" X = df.drop('species', axis=1)
y = df['species'] """

X = df.drop('taxonomy', axis = 1)
X = X[['lat', 'lon', 'geoaltitude', 'velocity', 'dropout_length']]
#X = X[['lat', 'lon', 'geoaltitude', 'velocity']]


y = df['taxonomy']

####################
# TRAIN TEST SPLIT #
####################

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=0)





######################################################################################
# 								TRAINING THE ALGORITHM								 #
######################################################################################

##########################################
# 			1. LINEAR KERNEL			 #
##########################################

start_time = time.time()

from sklearn.svm import SVC
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

print("--- %s seconds ---" % (time.time() - start_time))

######################
# MAKING PREDICTIONS #
######################

#y_pred = svclassifier.predict(X_test)

y_pred = clf.predict(X_test)

accuracy = clf.score(X_test, y_test) * 100



############################
# EVALUATING THE ALGORITHM #
############################

from sklearn.metrics import classification_report, confusion_matrix
print("\n########################################\n#\t\t\t1. LINEAR KERNEL\t\t\t#\n########################################")
print("Confusion Matrix:\n{}".format(confusion_matrix(y_test, y_pred, labels = clf.classes_)))
print("Classification Report:\n{}".format(classification_report(y_test, y_pred)))




from sklearn.metrics import ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred, labels = clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = clf.classes_)
disp.plot()
disp.ax_.set_title("Linear Kernel SVM Classifier\nAccuracy: {:.4f}%".format(accuracy))
plt.show()

exit(0)






##############################################
# 			2. POLYNOMIAL KERNEL			 #
##############################################

# from sklearn.svm import SVC
# svclassifier = SVC(kernel='poly', degree=8)
# svclassifier.fit(X_train, y_train)

start_time = time.time()

from sklearnex import patch_sklearn
patch_sklearn(["SVC"])
from sklearn.svm import SVC
clf = SVC(kernel='poly', degree=8)

clf.fit(X_train, y_train)

print("--- %s seconds ---" % (time.time() - start_time))

######################
# MAKING PREDICTIONS #
######################

#y_pred = svclassifier.predict(X_test)

y_pred = clf.predict(X_test)




############################
# EVALUATING THE ALGORITHM #
############################

from sklearn.metrics import classification_report, confusion_matrix
print("\n############################################\n#\t\t\t2. POLYNOMIAL KERNEL\t\t\t#\n############################################")
print("Confusion Matrix:\n{}".format(confusion_matrix(y_test, y_pred, labels = clf.classes_)))
print("Classification Report:\n{}".format(classification_report(y_test, y_pred)))




from sklearn.metrics import ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred, labels = clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = clf.classes_)
disp.plot()
plt.show()

exit(0)


##########################################
# 			3. GAUSSIAN KERNEL			 #
##########################################

from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)


######################
# MAKING PREDICTIONS #
######################

y_pred = svclassifier.predict(X_test)



############################
# EVALUATING THE ALGORITHM #
############################

from sklearn.metrics import classification_report, confusion_matrix
print("\n############################################\n# \t\t\t\t3. GAUSSIAN KERNEL\t\t\t\t #\n############################################")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))




##########################################
# 			4. SIGMOID KERNEL			 #
##########################################

from sklearn.svm import SVC
svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(X_train, y_train)


######################
# MAKING PREDICTIONS #
######################

y_pred = svclassifier.predict(X_test)



############################
# EVALUATING THE ALGORITHM #
############################

from sklearn.metrics import classification_report, confusion_matrix
print("\n#########################################\n# \t\t\t\t4. SIGMOID KERNEL\t\t\t\t #\n#########################################")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))