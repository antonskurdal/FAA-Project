# Original Code Source:
# https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/

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

#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
#colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Create an object called iris with the iris data
iris = load_iris()

# Create a dataframe with the four feature variables
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add a new column with the species names, this is what we are going to try to predict
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# View the top 5 rows
print(df.head())



#################
# PREPROCESSING #
#################

X = df.drop('species', axis=1)
y = df['species']



####################
# TRAIN TEST SPLIT #
####################

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)





######################################################################################
# 								TRAINING THE ALGORITHM								 #
######################################################################################

##############################################
# 			1. POLYNOMIAL KERNEL			 #
##############################################

from sklearn.svm import SVC
svclassifier = SVC(kernel='poly', degree=8)
svclassifier.fit(X_train, y_train)


######################
# MAKING PREDICTIONS #
######################

y_pred = svclassifier.predict(X_test)



############################
# EVALUATING THE ALGORITHM #
############################

from sklearn.metrics import classification_report, confusion_matrix
print("\n############################################\n# \t\t\t\t1. POLYNOMIAL KERNEL\t\t\t\t #\n############################################")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))




##########################################
# 			2. GAUSSIAN KERNEL			 #
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
print("\n############################################\n# \t\t\t\t2. GAUSSIAN KERNEL\t\t\t\t #\n############################################")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))




##########################################
# 			3. SIGMOID KERNEL			 #
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
print("\n#########################################\n# \t\t\t\t3. SIGMOID KERNEL\t\t\t\t #\n#########################################")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))