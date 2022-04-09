#!/usr/bin/env python

"""Random Forest Classifier
    
    Original Code Source:
    https://chrisalbon.com/code/machine_learning/trees_and_forests/random_forest_classifier_example/
    
    description
"""
# from sklearnex import patch_sklearn
# patch_sklearn(["RandomForestClassifier"])
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import time
from pathlib import Path
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

__author__ = "Anton Skurdal"
__copyright__ = "Copyright 2022, The FAA Project"
__credits__ = ["Anton Skurdal"]
__license__ = "GPL"
__version__ = "1.5"
__maintainer__ = "Anton Skurdal"
__email__ = "antonskurdal@gmail.com"
__status__ = "Development"


#################
# PRELIMINARIES #
#################
# Set random seed
import time
np.random.seed(1)
#np.random.seed(int(time.time()))

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

print(df)
print(df.shape[0])
print(df['taxonomy'].value_counts())

#time.sleep(2)

fig = px.strip(df, x="taxonomy", y="dropout_length", color="taxonomy", stripmode='overlay')
#fig.show()



""" hue_order = ['normal', 'noise', 'erroneous', 'dropout']
sns.set(rc={'figure.figsize':(12,6)})
#fig, ax = plt.subplots(figsize=(10, 8))
sns.set_theme(style="whitegrid")
ax = sns.stripplot(data = df, x = "dropout_length", y = "taxonomy", hue_order=hue_order, jitter=0.4)
#ax.set_xscale("log")

plt.show() """










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

df['is_train'] = np.random.uniform(0, 1, len(df)) <= 0.6

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
print(list(y))
print(np.unique(y))



######################################
# TRAIN THE RANDOM FOREST CLASSIFIER #
######################################

""" # Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_jobs=2, random_state=0)

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
clf.fit(train[features], y) """

start_time = time.time()

clf = RandomForestClassifier(n_jobs = 10, random_state=0)
clf.fit(train[features], y)

print("--- %s seconds ---" % (time.time() - start_time))

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

y_test = pd.factorize(test['taxonomy'])[0]

from sklearn.metrics import classification_report
print("Classification Report:\n{}".format(classification_report(y_true, y_pred)))


accuracy = 100 * metrics.accuracy_score(y_true, y_pred)
print("Accuracy: {:.4f}".format(metrics.accuracy_score(y_true, y_pred)))
print("PREDS: {}".format(preds))
print("TEST TAX: {}".format(test['taxonomy']))
print("CLF CLASSES: {}".format(clf.classes_))

codes_test, _y_test = pd.factorize(test['taxonomy'])
#y = pd.factorize(train['taxonomy'])[0]
target_names_test = _y_test.take(codes_test).unique()
print("target names test: {}".format(target_names_test))

test['random_forest_prediction'] = list(preds)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(test['taxonomy'], preds, labels = target_names_test[clf.classes_])
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = target_names_test[clf.classes_])
disp.plot()
disp.ax_.set_title("Random Forest Classifier\nAccuracy: {:.4f}%".format(accuracy))
plt.show()










###################################################################################################
# library
import matplotlib.pyplot as plt
 
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10,6))
# create data
names = test['taxonomy'].value_counts().index
names_pred = test['random_forest_prediction'].value_counts().index

size = test['taxonomy'].value_counts()
size_pred = test['random_forest_prediction'].value_counts()
 
wedgeprops = {'width':0.5, 'edgecolor':'white', 'linewidth':3}
wedgeprops_pred = {'width':0.5, 'edgecolor':'white', 'linewidth':3}
# Create a circle at the center of the plot
my_circle = plt.Circle( (0,0), 0.7, color='white')
my_circle_preds = plt.Circle( (0,0), 0.7, color='white')

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%\n({v:d})'.format(p=pct,v=val)
    return my_autopct


# Actual Labels
wedgeprops, texts, autotexts = ax0.pie(
    size,
    labels=names, 
    colors=['tab:orange','tab:green','tab:blue','tab:red'], 
    wedgeprops=wedgeprops, autopct=make_autopct(size), 
    pctdistance = 0.75, 
    labeldistance = None
    )
plt.setp(autotexts, **{'color':'black', 'weight':'bold', 'fontsize':12})
ax0.legend(frameon = False, loc = 'center')
ax0.set_title("Actual Labels (Test Set)")


# Random Forest Predicted Labels
wedgeprops_pred, texts_pred, autotexts_pred = ax1.pie(
    size_pred,
    labels=names_pred, 
    colors=['tab:orange','tab:green','tab:blue','tab:red'], 
    wedgeprops=wedgeprops_pred, autopct=make_autopct(size_pred), 
    pctdistance = 0.75, 
    labeldistance = None
    )
ax1.text(0, 0, "Accuracy:\n{:.4f}%".format(accuracy), ha='center', va='center', fontsize=14)
plt.setp(autotexts_pred, **{'color':'black', 'weight':'bold', 'fontsize':12})
ax1.set_title("Random Forest Predicted Labels (Test Set)")


# Show the graph
plt.tight_layout()
plt.show()
###################################################################################################















import seaborn as sns
import matplotlib.pyplot as plt
hue_order = ['normal', 'erroneous', 'noise', 'dropout']

plt.figure(1)
sns.scatterplot(data = test, x = "time", y = "dropout_length", hue = "random_forest_prediction", hue_order=hue_order)
plt.title("Random Forest Predicted Taxonomy/Labels\nAccuracy: {:.4f}%".format(100 * (metrics.accuracy_score(y_true, y_pred))))
#plt.show()

plt.figure(2)
hue_order = ['normal', 'erroneous', 'noise', 'dropout']

sns.scatterplot(data = test, x = "time", y = "dropout_length", hue = "taxonomy", hue_order=hue_order)
plt.title("Actual Taxonomy/Labels")
plt.show()



from pandas.plotting import parallel_coordinates

plt.figure(3)
pfeatures = features.append("taxonomy")
p = df[features]
#p = p[p['taxonomy'] == "dropout"]
parallel_coordinates(p, "taxonomy", colormap = plt.get_cmap("Set2"))





#plt.show()



#exit(0)

""" # Import the library
import plotly.express as px

# Load the iris dataset provided by the library
#df = px.data.iris()

df['labels_fact'] = pd.factorize(df['taxonomy'])[0]
pfeatures = features.append('labels_fact')
p = df[features]
p = p[p['taxonomy'] == "dropout"]
# Create the chart:
fig = px.parallel_coordinates(
    p, 
    #color = "labels_fact", 
    labels={"lat": "Lat", "lon": "Lon", "velocity": "Velocity", "geoaltitude": "Altitude (geo)"},
    #color_continuous_scale=px.colors.diverging.Tealrose,
    #color_continuous_midpoint=2)
)

# Hide the color scale that is useless in this case
fig.update_layout(coloraxis_showscale=False)

# Show the plot
fig.show() """
















# Import the library
import plotly.express as px

# Load the iris dataset provided by the library
#df = px.data.iris()

codes, _y = pd.factorize(df['taxonomy'])

print("CODES:\n{}".format(_y.take(codes).unique()))

df['labels_fact'] = pd.factorize(df['taxonomy'])[0]
pfeatures = features.append('labels_fact')
p = df[features]
# Create the chart:
fig = px.parallel_coordinates(
    p, 
    color = "labels_fact", 
    labels={"lat": "Lat", "lon": "Lon", "velocity": "Velocity", "geoaltitude": "Altitude (geo)", "labels_fact": "Class", },
    #color_continuous_scale=px.colors.diverging.Tealrose,
    #color_continuous_midpoint=2)
)

# Hide the color scale that is useless in this case
fig.update_layout(coloraxis_showscale=False)

# Show the plot
#fig.write_html("plotly_test.html")
fig.show()

















""" # Import the library
import plotly.express as px

# Load the iris dataset provided by the library
df = px.data.iris()

# Create the chart:
fig = px.parallel_coordinates(
    df, 
    color="species_id", 
    labels={"species_id": "Species","sepal_width": "Sepal Width", "sepal_length": "Sepal Length", "petal_width": "Petal Width", "petal_length": "Petal Length", },
    color_continuous_scale=px.colors.diverging.Tealrose,
    color_continuous_midpoint=2)

# Hide the color scale that is useless in this case
fig.update_layout(coloraxis_showscale=False)

# Show the plot
fig.show() """








exit(0)