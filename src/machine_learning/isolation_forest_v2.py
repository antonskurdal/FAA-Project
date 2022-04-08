# Original Code Source:
# https://scikit-learn.org/stable/auto_examples/ensemble/plot_isolation_forest.html


import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(42)

import pandas as pd
from pathlib import Path

#############
# LOAD DATA #
#############
directory = Path.cwd() / "data" / "ML-datasets" / "RandomForest"
print(directory.glob('**/*'))
files = [f for f in directory.glob('**/*.csv')]
print(files)
#print([f.name for f in directory.glob('**/*.csv')])
df = pd.concat(map(pd.read_csv, files), ignore_index = True)
print(df.columns)
df = df.drop('Unnamed: 0', axis = 1)
df = df.dropna(axis = 0, how = 'any', subset = ['lat', 'lon', 'geoaltitude', 'velocity', 'dropout_length'])
print(df)
print("\nDataframe Size: {}".format(df.shape[0]))
print("\nValue Counts:\n{}".format(df['taxonomy'].value_counts()))

#features = ['lat', 'lon', 'geoaltitude', 'velocity', 'dropout_length']
features = ['lat', 'lon', 'geoaltitude', 'velocity']

contamination = float(0.028)


#model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1),max_features=1.0)
#model=IsolationForest(n_estimators=50, max_samples='auto', contamination='auto',max_features=1.0)
model=IsolationForest(n_estimators=50, max_samples='auto', contamination=contamination,max_features=1.0)
model.fit(df[['dropout_length']])

df['isolation_forest_scores']=model.decision_function(df[['dropout_length']])
df['isolation_forest_anomaly']=model.predict(df[['dropout_length']])
df['isolation_forest_prediction'] = df['isolation_forest_anomaly'].map({1: "normal", -1: "anomaly"})


print(df.head(20))

outlier_counter = len(df[df['taxonomy'] == "dropout"])# + len(df[df['taxonomy'] == "noise"])
print("Actual Outlier (Dropout or Noise) Count: {}".format(outlier_counter))
print("Predicated Anomaly Count: {}".format(list(df['isolation_forest_anomaly']).count(-1)))

print("Accuracy percentage: {:.4f}%".format(100*list(df['isolation_forest_anomaly']).count(-1)/(outlier_counter)))


import seaborn as sns
plt.figure(1)
sns.scatterplot(data = df, x = "time", y = "dropout_length", hue = "isolation_forest_prediction", palette=['tab:blue', 'tab:red'])
plt.title("Isolation Forest Prediction (Anomaly: True or False)\nAccuracy: {:.4f}%".format(100*list(df['isolation_forest_anomaly']).count(-1)/(outlier_counter)))
#plt.show()

plt.figure(2)
hue_order = ['normal', 'erroneous', 'noise', 'dropout']

sns.scatterplot(data = df, x = "time", y = "dropout_length", hue = "taxonomy", hue_order=hue_order)
plt.title("Actual Taxonomy/Labels")
plt.show()

exit(0)













#################################
# CREATE TRAINING AND TEST DATA #
#################################
""" # Generate train data
X = 0.3 * rng.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
# Generate some regular novel observations
X = 0.3 * rng.randn(20, 2)
X_test = np.r_[X + 2, X - 2]
# Generate some abnormal novel observations
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2)) """



codes, y = pd.factorize(df['taxonomy'])
#print(y.take(codes).unique())
#print(df['taxonomy'].unique())


#################
# FIT THE MODEL #
#################
""" # fit the model
clf = IsolationForest(max_samples=100, random_state=rng)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers) """

#model = IsolationForest(max_samples=100, random_state=rng)
model = IsolationForest(n_jobs = 10, n_estimators = 1000, contamination=contamination)
model.fit(df[features])

df['isolation_forest'] = pd.Series(model.predict(df[features]))

# Make predictions easier to read for humans
df['isolation_forest'] = df['isolation_forest'].map({1: 0, -1: 1})

preds = df['isolation_forest'].value_counts()
#print(df['isolation_forest'].value_counts())
print("\nPrediction Counts:\n{}".format(preds))
print("\nAnomaly Percentage: {:4f}%".format((preds[1] / preds[0]) * 100))

exit(0)
##################
# PLOT THE MODEL #
##################
# plot the line, the samples, and the nearest vectors to the plane
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("IsolationForest")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c="white", s=20, edgecolor="k")
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c="green", s=20, edgecolor="k")
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c="red", s=20, edgecolor="k")
plt.axis("tight")
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend(
    [b1, b2, c],
    ["training observations", "new regular observations", "new abnormal observations"],
    loc="upper left",
)
plt.show()


exit(0)








#################
# PRELIMINARIES #
#################

# Load the library with the iris dataset
from sklearn.datasets import load_iris

# Load scikit's random forest classifier library
from sklearn.ensemble import IsolationForest

# Load pandas
import pandas as pd

# Load numpy
import numpy as np

# Set random seed
np.random.seed(0)

from pathlib import Path
from sklearn import metrics

#############
# LOAD DATA #
#############
#iris = load_iris()

""" # Create an object called iris with the iris data
iris = load_iris()

# Create a dataframe with the four feature variables
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# View the top 5 rows
df.head()


# Add a new column with the species names, this is what we are going to try to predict
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# View the top 5 rows
df.head() """




directory = Path.cwd() / "data" / "ML-datasets" / "RandomForest"
print(directory.glob('**/*'))

files = [f for f in directory.glob('**/*.csv')]
print(files)
#print([f.name for f in directory.glob('**/*.csv')])

df = pd.concat(map(pd.read_csv, files), ignore_index = True)
print(df.columns)
df = df.drop('Unnamed: 0', axis = 1)

df = df.dropna(axis = 0, how = 'any', subset = ['lat', 'lon', 'geoaltitude', 'velocity', 'dropout_length'])

print(df)
print(df.shape[0])


#################################
# CREATE TRAINING AND TEST DATA #
#################################

""" # Create a new column that for each row, generates a random number between 0 and 1, and
# if that value is less than or equal to .75, then sets the value of that cell as True
# and false otherwise. This is a quick and dirty way of randomly assigning some rows to
# be used as the training data and some as the test data.
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75

# View the top 5 rows
df.head()


# Create two new dataframes, one with the training rows, one with the test rows
train, test = df[df['is_train']==True], df[df['is_train']==False]


# Show the number of observations for the test and training dataframes
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test)) """

df['is_train'] = np.random.uniform(0, 1, len(df)) <= 0.75

train, test = df[df['is_train'] == True], df[df['is_train'] == False]


###################
# PREPROCESS DATA #
###################

""" # Create a list of the feature column's names
features = df.columns[:4]

# View features
#features
print(features)


# train['species'] contains the actual species names. Before we can use it,
# we need to convert each species name into a digit. So, in this case there
# are three species, which have been coded as 0, 1, or 2.
y = pd.factorize(train['species'])[0]

# View target
#y
print(y) """


features = ['lat', 'lon', 'geoaltitude', 'velocity', 'dropout_length']
#features = ['lat', 'lon', 'geoaltitude', 'velocity']
print(features)

codes, _y = pd.factorize(train['taxonomy'])
y = pd.factorize(train['taxonomy'])[0]
#print(list(y))
print(np.unique(y))




######################################
# TRAIN THE RANDOM FOREST CLASSIFIER #
######################################

""" # Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_jobs=2, random_state=0)

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
clf.fit(train[features], y) """

clf = IsolationForest(n_jobs = 10, random_state=0)

clf.fit(train[features], y)


#################################
# APPLY CLASSIFIER TO TEST DATA #
#################################

""" # Apply the Classifier we trained to the test data (which, remember, it has never seen before)
clf.predict(test[features])


# View the predicted probabilities of the first 10 observations
clf.predict_proba(test[features])[0:10] """

clf.predict(test[features])

clf.predict_proba(test[features])[0:10]


#######################
# EVALUATE CLASSIFIER #
#######################

""" # Create actual english names for the plants for each predicted plant class
preds = iris.target_names[clf.predict(test[features])]
#print(iris.target_names)


# View the PREDICTED species for the first five observations
#preds[0:5]
print(preds[0:5])


# View the ACTUAL species for the first five observations
test['species'].head() """


#target_names = ['normal', 'noise', 'erroneous', 'dropout']

#print(_y.take(codes).unique())
target_names = _y.take(codes).unique()
#print(target_names)


preds = target_names[clf.predict(test[features])]
print(preds[0:5])
print(test['taxonomy'].head())


#############################
# CREATE A CONFUSION MATRIX #
#############################

""" # Create confusion matrix
pd.crosstab(test['species'], preds, rownames=['Actual Species'], colnames=['Predicted Species']) """

print(pd.crosstab(test['taxonomy'], preds, rownames=['Actual Taxonomy'], colnames=['Predicted Taxonomy']))



###########################
# VIEW FEATURE IMPORTANCE #
###########################

""" # View a list of the features and their importance scores
#list(zip(train[features], clf.feature_importances_))
print(list(zip(train[features], clf.feature_importances_))) """

print(list(zip(train[features], clf.feature_importances_)))


##################
# EXTRA ANALYSIS #
##################
y_true = list(test['taxonomy'])
y_pred = preds
print(metrics.accuracy_score(y_true, y_pred))