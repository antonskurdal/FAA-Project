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

training_data = pd.DataFrame()

training_data['test_1'] = [0.3051,0.4949,0.6974,0.3769,0.2231,0.341,0.4436,0.5897,0.6308,0.5]
training_data['test_2'] = [0.5846,0.2654,0.2615,0.4538,0.4615,0.8308,0.4962,0.3269,0.5346,0.6731]
training_data['outcome'] = ['win','win','win','win','win','loss','loss','loss','loss','loss']

#training_data.head()
print(training_data.head())


#################
# PLOT THE DATA #
#################

#seaborn.lmplot('test_1', 'test_2', data=training_data, fit_reg=False,hue="outcome", scatter_kws={"marker": "D","s": 100})
seaborn.lmplot(x='test_1', y='test_2', data=training_data, fit_reg=False,hue="outcome", scatter_kws={"marker": "D","s": 100})
plt.show()



###############################
# CONVERT DATA INTO NP.ARRAYS #
###############################

#X = training_data.as_matrix(columns=['test_1', 'test_2'])
X = training_data[['test_1', 'test_2']].to_numpy()
y = np.array(training_data['outcome'])



#####################
# TRAIN THE LEARNER #
#####################

clf = neighbors.KNeighborsClassifier(3, weights = 'uniform')
trained_model = clf.fit(X, y)



##########################
# VIEW THE MODEL'S SCORE #
##########################

#trained_model.score(X, y)
print(trained_model.score(X, y))



#########################################
# APPLY THE LEARNER TO A NEW DATA POINT #
#########################################

# Create a new observation with the value of the first independent variable, 'test_1', as .4 
# and the second independent variable, test_1', as .6 
x_test = np.array([[.4,.6]])


# Apply the learner to the new, unclassified observation.
#trained_model.predict(x_test)
print(trained_model.predict(x_test))


#trained_model.predict_proba(x_test)
print(trained_model.predict_proba(x_test))