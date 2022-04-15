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
#np.random.seed(1)
np.random.seed(int(time.time()))


#############
# LOAD DATA #
#############
# Set up directory
parent_directory = Path("D:/#FAA UAS Project/OpenSky WEEK/Individual Aircraft/batch_0/output")

agg_data = False
if(agg_data):
    
    #directory = Path.cwd() / "data" / "ML-datasets" / "RandomForest"

    # Open directory
    extensions = ('*.parquet', '*.csv')
    file_count = 0
    for ext in extensions:
        for file in parent_directory.rglob(ext):
            #print(file.name)
            file_count += 1
    print("File Count: {}".format(file_count))

    # Concatenate all files into a dataframe
    parquet_files = [f for f in parent_directory.rglob('*.parquet')]
    csv_files = [f for f in parent_directory.rglob('*.csv')]
    parquet_df = pd.concat(map(pd.read_parquet, parquet_files), ignore_index = True)
    csv_df = pd.concat(map(pd.read_csv, csv_files), ignore_index = True)
    df = pd.concat([parquet_df, csv_df])

    # Remove irrelevant columns
    relevant_columns = ['time', 'taxonomy', 'icao24', 'lat', 'lon', 'geoaltitude', 'velocity', 'lastcontact', 'dropout_length']
    df = df[relevant_columns]
    print("Dataset Columns: {}".format(list(df.columns)))

    # Drop invalid data
    #df = df.dropna(axis = 0, how = 'any', subset = ['lat', 'lon', 'geoaltitude', 'velocity', 'dropout_length'])
    df = df.dropna(axis = 0, how = 'any', subset = ['lat', 'lon', 'geoaltitude', 'velocity', 'dropout_length', 'lastcontact'])

    # Drop duplicates
    df = df.drop_duplicates()
    print("Number of Unique Aircraft: {}".format(len(df['icao24'].unique())))
    print("Data Points Count: {}".format(df.shape[0]))


    # Save aggregated file
    print(parent_directory)
    f = Path("#RF2_AGG.csv")
    df.to_csv(parent_directory / f)

    exit()

else:
    file = parent_directory / "#RF2_AGG.csv"
    #file = Path("D:/#FAA UAS Project/OpenSky WEEK/Individual Aircraft/batch_0/output/#RF2_AGG.csv")
    df = pd.read_csv(file)



# Drop erroneous
#df = df[df['taxonomy'] != 'erroneous']

# # Drop noise
# df = df[df['taxonomy'] != 'noise']



print("Taxonomy Counts:\n{}\n".format(df['taxonomy'].value_counts()))

#time.sleep(2)

#fig = px.strip(df, x="taxonomy", y="dropout_length", color="taxonomy", stripmode='overlay')
#fig.show()

#################################
# CREATE TRAINING AND TEST DATA #
#################################

train_percentage = 0.7
df['is_train'] = np.random.uniform(0, 1, len(df)) <= train_percentage
train, test = df[df['is_train'] == True], df[df['is_train'] == False]

print("\nTraining Set Size: {:.4f}% ({}/{})".format(((train.shape[0]/df.shape[0]) * 100), train.shape[0], df.shape[0]))
print("Training Taxonomy Counts:\n{}\n".format(train['taxonomy'].value_counts()))

print("Testing Set Size: {:.4f}% ({}/{})".format(((test.shape[0]/df.shape[0]) * 100), test.shape[0], df.shape[0]))
print("Testing Taxonomy Counts:\n{}\n".format(test['taxonomy'].value_counts()))

#print("Testing Set Size: {}% ({}/{})\n".format(train.shape[0], test.shape[0]))


# Define base dataset and features
features = ['lat', 'lon', 'geoaltitude', 'velocity', 'lastcontact']#, 'dropout_length', 'lastcontact']
X = df[features]
y = df['taxonomy']

# Train/Test Split
# X_train -> training features
# y_train -> training labels
# X_test -> testing features
# y_test -> testing labels
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print("\n")
_df = X_train
print("X_train Set Size: {:.4f}% ({}/{})".format(((_df.shape[0]/df.shape[0]) * 100), _df.shape[0], _df.shape[0]))
_df = y_train
print("y_train Taxonomy Counts:\n{}\n".format(_df.value_counts()))
_df = X_test
print("X_test Set Size: {:.4f}% ({}/{})".format(((_df.shape[0]/df.shape[0]) * 100), _df.shape[0], _df.shape[0]))
_df = y_test
print("y_test Taxonomy Counts:\n{}\n".format(_df.value_counts()))

# Base model
print("Base Model Starting...")
start_time = time.time()
rf = RandomForestClassifier(n_jobs = 16, n_estimators = 20, random_state = 42)
rf.fit(X_train, y_train)
print("--- Elapsed Time: %s seconds ---\n" % (time.time() - start_time))


# Predict test set
y_pred = rf.predict(X_test)
y_true = y_test

# Show classification report
from sklearn.metrics import classification_report
report = classification_report(y_true, y_pred, labels = rf.classes_)
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
disp.ax_.set_title("Random Forest Classifier\nAccuracy: {:.4f}%".format(accuracy*100))
plt.show()

























exit()
###################
# PREPROCESS DATA #
###################


features = ['lat', 'lon', 'geoaltitude', 'velocity', 'dropout_length', 'lastcontact']
#features = ['lat', 'lon', 'geoaltitude', 'velocity', 'lastcontact']

#features = ['lat', 'lon', 'geoaltitude', 'velocity', 'dropout_length']
#features = ['lat', 'lon', 'geoaltitude', 'velocity']
print("Features: {}".format(features))

codes, _y = pd.factorize(train['taxonomy'])
y = pd.factorize(train['taxonomy'])[0]
# print(list(y))
# print(np.unique(y))


###############################
# Hyperparameter Optimization #
###############################

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy


# Organize the data
train_features = train[features]
train_labels = train['taxonomy']
codes_train, uniques_train = pd.factorize(train_labels)
train_labels = codes_train
#print("Train Features Columns: {}".format(list(train_features.columns)))
#print("Train Labels: {}".format(list(train_labels)))
test_features = test[features]
test_labels = test['taxonomy']
codes_test, uniques_test = pd.factorize(test_labels)
test_labels = codes_test
#print("Test Features Columns: {}".format(list(test_features.columns)))


# Base model
print("Base Model Starting...")
start_time = time.time()
base_model = RandomForestClassifier(n_jobs = 16, n_estimators = 10, random_state = 42)
base_model.fit(train_features, train_labels)
print("--- Elapsed Time: %s seconds ---" % (time.time() - start_time))
#base_accuracy = evaluate(base_model, test_features, test_labels)
base_accuracy = metrics.accuracy_score(test_labels, base_model.predict(test_features))

from sklearn.metrics import classification_report
print("Classification Report:\n{}".format(classification_report(test_labels, base_model.predict(test_features))))


# Evaluating the Algorithm
from sklearn import metrics
print("\n")
print("Accuracy: {:.4f}%".format((metrics.accuracy_score(test_labels, base_model.predict(test_features))*100)))
print('Mean Absolute Error:', metrics.mean_absolute_error(test_labels, base_model.predict(test_features)))
print('Mean Squared Error:', metrics.mean_squared_error(test_labels, base_model.predict(test_features)))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labels, base_model.predict(test_features))))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(test_labels, base_model.predict(test_features), labels = base_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = uniques_test[base_model.classes_])
disp.plot()
disp.ax_.set_title("Random Forest Classifier\nAccuracy: {:.4f}%".format(base_accuracy*100))
plt.show()












exit()

# GridSearchCV
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    # 'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    # 'min_samples_leaf': [3, 4, 5],
    # 'min_samples_split': [8, 10, 12],
    #'n_estimators': [100, 200, 300, 1000]
    'n_estimators': [10, 50, 100]
}

# Create a based model
rf = RandomForestClassifier()# Instantiate the grid search model
print("GridSearchCV Starting...")
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = 16, verbose = 2)



# Fit the grid search to the data
grid_search.fit(train_features, train_labels)
print("--- Elapsed Time: %s seconds ---" % (time.time() - start_time))

grid_search.best_params_
{
    'bootstrap': True,
    'max_depth': 80,
    'max_features': 3,
    'min_samples_leaf': 5,
    'min_samples_split': 12,
    'n_estimators': 100
}
best_grid = grid_search.best_estimator_
#grid_accuracy = evaluate(best_grid, test_features, test_labels)
grid_accuracy = metrics.accuracy_score(test_labels, best_grid.predict(test_features))
# Evaluating the Algorithm
from sklearn import metrics
print("\n")
print("Accuracy: {:.4f}".format(metrics.accuracy_score(test_labels, best_grid.predict(test_features))))
print('Mean Absolute Error:', metrics.mean_absolute_error(test_labels, best_grid.predict(test_features)))
print('Mean Squared Error:', metrics.mean_squared_error(test_labels, best_grid.predict(test_features)))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labels, best_grid.predict(test_features))))

print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))
exit()
"""
Training Set Size: 60.0080% (1942929/3237783)
Testing Set Size: 39.9920% (1294854/3237783)

Features: ['lat', 'lon', 'geoaltitude', 'velocity', 'dropout_length', 'lastcontact']
Base Model Starting...
--- Elapsed Time: 4.925688028335571 seconds ---


Accuracy: 0.3906
Mean Absolute Error: 1.2160799595939003
Mean Squared Error: 2.4294924369851736
Root Mean Squared Error: 1.558682917396984
GridSearchCV Starting...
Fitting 3 folds for each of 6 candidates, totalling 18 fits
[CV] END ....bootstrap=True, max_features=2, n_estimators=10; total time=  38.0s
[CV] END ....bootstrap=True, max_features=3, n_estimators=10; total time=  49.2s
[CV] END ....bootstrap=True, max_features=3, n_estimators=10; total time=  51.8s
[CV] END ....bootstrap=True, max_features=3, n_estimators=10; total time=  52.1s
[CV] END ....bootstrap=True, max_features=2, n_estimators=50; total time= 2.6min
[CV] END ....bootstrap=True, max_features=2, n_estimators=50; total time= 2.6min
[CV] END ....bootstrap=True, max_features=2, n_estimators=50; total time= 2.7min
[CV] END ....bootstrap=True, max_features=3, n_estimators=50; total time= 3.4min
[CV] END ....bootstrap=True, max_features=3, n_estimators=50; total time= 3.5min
[CV] END ....bootstrap=True, max_features=3, n_estimators=50; total time= 3.5min
--- Elapsed Time: 539.2939231395721 seconds ---


Accuracy: 0.3921
Mean Absolute Error: 1.2143137373016573
Mean Squared Error: 2.427068225452445
Root Mean Squared Error: 1.5579050758799282
Improvement of 0.37%.
[CV] END ...bootstrap=True, max_features=2, n_estimators=100; total time= 4.4min
[CV] END ...bootstrap=True, max_features=2, n_estimators=100; total time= 4.4min
[CV] END ...bootstrap=True, max_features=2, n_estimators=100; total time= 4.5min
[CV] END ...bootstrap=True, max_features=3, n_estimators=100; total time= 5.6min
[CV] END ....bootstrap=True, max_features=2, n_estimators=10; total time=  36.1s
[CV] END ...bootstrap=True, max_features=3, n_estimators=100; total time= 5.2min
[CV] END ....bootstrap=True, max_features=2, n_estimators=10; total time=  37.9s
[CV] END ...bootstrap=True, max_features=3, n_estimators=100; total time= 5.3min """


######################################
# TRAIN THE RANDOM FOREST CLASSIFIER #
######################################

start_time = time.time()

#clf = RandomForestClassifier(n_jobs = 10, random_state=0)
clf = RandomForestClassifier(n_jobs = 10, n_estimators=100)#, random_state=0, )

clf.fit(train[features], y)

print("--- %s seconds ---" % (time.time() - start_time))

#################################
# APPLY CLASSIFIER TO TEST DATA #
#################################

clf.predict(test[features])

clf.predict_proba(test[features])[0:10]


#######################
# EVALUATE CLASSIFIER #
#######################

#target_names = ['normal', 'noise', 'erroneous', 'dropout']

#print(_y.take(codes).unique())
target_names = _y.take(codes).unique()
#print(target_names)


preds = target_names[clf.predict(test[features])]
# print(preds[0:5])
# print("TEST TAX HEAD")
# print(test['taxonomy'].head())
# print("CLF PREDS")
# print(clf.predict(test[features]))

# Evaluating the Algorithm
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(pd.factorize(test['taxonomy'])[0], clf.predict(test[features])))
print('Mean Squared Error:', metrics.mean_squared_error(pd.factorize(test['taxonomy'])[0], clf.predict(test[features])))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(pd.factorize(test['taxonomy'])[0], clf.predict(test[features]))))

#############################
# CREATE A CONFUSION MATRIX #
#############################


print(pd.crosstab(test['taxonomy'], preds, rownames=['Actual Taxonomy'], colnames=['Predicted Taxonomy']))




###########################
# VIEW FEATURE IMPORTANCE #
###########################

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
print("Accuracy: {:.4f}\n\n\n".format(metrics.accuracy_score(y_true, y_pred)))
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


exit()







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