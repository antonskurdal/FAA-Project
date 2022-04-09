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
from sklearn import neighbors
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn



##################
# CREATE DATASET #
##################

""" training_data = pd.DataFrame()

training_data['test_1'] = [0.3051,0.4949,0.6974,0.3769,0.2231,0.341,0.4436,0.5897,0.6308,0.5]
training_data['test_2'] = [0.5846,0.2654,0.2615,0.4538,0.4615,0.8308,0.4962,0.3269,0.5346,0.6731]
training_data['outcome'] = ['win','win','win','win','win','loss','loss','loss','loss','loss']

#training_data.head()
print(training_data.head()) """


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


df['is_train'] = np.random.uniform(0, 1, len(df)) <= 0.6

train, test = df[df['is_train'] == True], df[df['is_train'] == False]




#################
# PLOT THE DATA #
#################

#seaborn.lmplot('test_1', 'test_2', data=training_data, fit_reg=False,hue="outcome", scatter_kws={"marker": "D","s": 100})
""" seaborn.lmplot(x='test_1', y='test_2', data=training_data, fit_reg=False,hue="outcome", scatter_kws={"marker": "D","s": 100})
plt.show() """
# hue_order = ['normal', 'erroneous', 'noise', 'dropout']
# seaborn.lmplot(x='time', y='dropout_length', data=train, fit_reg=False,hue="taxonomy", scatter_kws={"marker": "D","s": 100}, hue_order=hue_order)
# plt.show()


###############################
# CONVERT DATA INTO NP.ARRAYS #
###############################

#X = training_data.as_matrix(columns=['test_1', 'test_2'])
""" X = training_data[['test_1', 'test_2']].to_numpy()
y = np.array(training_data['outcome']) """

features = ['lat', 'lon', 'geoaltitude', 'velocity', 'dropout_length']
#features = ['lat', 'lon', 'geoaltitude', 'velocity']
X = train[features].to_numpy()


y = np.array(train['taxonomy'])
# codes, _y = pd.factorize(train['taxonomy'])
# y = pd.factorize(train['taxonomy'])[0]


#####################
# TRAIN THE LEARNER #
#####################

""" clf = neighbors.KNeighborsClassifier(3, weights = 'uniform')
trained_model = clf.fit(X, y) """
n_neighbors = 3
clf = neighbors.KNeighborsClassifier(n_neighbors, weights = 'uniform')
trained_model = clf.fit(X, y)



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