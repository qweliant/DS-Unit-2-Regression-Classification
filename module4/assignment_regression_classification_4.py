# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'module4'))
	print(os.getcwd())
except:
	pass
#%%
from IPython import get_ipython

#%% [markdown]
# Lambda School Data Science, Unit 2: Predictive Modeling
# 
# # Regression & Classification, Module 4
# 
# 
# ## Assignment
# 
# - [ ] Watch Aaron's [video #1](https://www.youtube.com/watch?v=pREaWFli-5I) (12 minutes) & [video #2](https://www.youtube.com/watch?v=bDQgVt4hFgY) (9 minutes) to learn about the mathematics of Logistic Regression.
# - [ ] [Sign up for a Kaggle account](https://www.kaggle.com/), if you don’t already have one. Go to our Kaggle InClass competition website. You will be given the URL in Slack. Go to the Rules page. Accept the rules of the competition.
# - [ ] Do train/validate/test split with the Tanzania Waterpumps data.
# - [ ] Begin with baselines for classification.
# - [ ] Use scikit-learn for logistic regression.
# - [ ] Get your validation accuracy score.
# - [ ] Submit your predictions to our Kaggle competition. (Go to our Kaggle InClass competition webpage. Use the blue **Submit Predictions** button to upload your CSV file. Or you can use the Kaggle API to submit your predictions.)
# - [ ] Commit your notebook to your fork of the GitHub repo.
# 
# ---
# 
# 
# ## Stretch Goals
# 
# - [ ] Add your own stretch goal(s) !
# - [ ] Clean the data. For ideas, refer to [The Quartz guide to bad data](https://github.com/Quartz/bad-data-guide),  a "reference to problems seen in real-world data along with suggestions on how to resolve them." One of the issues is ["Zeros replace missing values."](https://github.com/Quartz/bad-data-guide#zeros-replace-missing-values)
# - [ ] Make exploratory visualizations.
# - [ ] Do one-hot encoding. For example, you could try `quantity`, `basin`, `extraction_type_class`, and more. (But remember it may not work with high cardinality categoricals.)
# - [ ] Do [feature scaling](https://scikit-learn.org/stable/modules/preprocessing.html).
# - [ ] Get and plot your coefficients.
# - [ ] Try [scikit-learn pipelines](https://scikit-learn.org/stable/modules/compose.html).
# 
# ---
# 
# ## Data Dictionary 
# 
# ### Features
# 
# Your goal is to predict the operating condition of a waterpoint for each record in the dataset. You are provided the following set of information about the waterpoints:
# 
# - `amount_tsh` : Total static head (amount water available to waterpoint)
# - `date_recorded` : The date the row was entered
# - `funder` : Who funded the well
# - `gps_height` : Altitude of the well
# - `installer` : Organization that installed the well
# - `longitude` : GPS coordinate
# - `latitude` : GPS coordinate
# - `wpt_name` : Name of the waterpoint if there is one
# - `num_private` :  
# - `basin` : Geographic water basin
# - `subvillage` : Geographic location
# - `region` : Geographic location
# - `region_code` : Geographic location (coded)
# - `district_code` : Geographic location (coded)
# - `lga` : Geographic location
# - `ward` : Geographic location
# - `population` : Population around the well
# - `public_meeting` : True/False
# - `recorded_by` : Group entering this row of data
# - `scheme_management` : Who operates the waterpoint
# - `scheme_name` : Who operates the waterpoint
# - `permit` : If the waterpoint is permitted
# - `construction_year` : Year the waterpoint was constructed
# - `extraction_type` : The kind of extraction the waterpoint uses
# - `extraction_type_group` : The kind of extraction the waterpoint uses
# - `extraction_type_class` : The kind of extraction the waterpoint uses
# - `management` : How the waterpoint is managed
# - `management_group` : How the waterpoint is managed
# - `payment` : What the water costs
# - `payment_type` : What the water costs
# - `water_quality` : The quality of the water
# - `quality_group` : The quality of the water
# - `quantity` : The quantity of water
# - `quantity_group` : The quantity of water
# - `source` : The source of the water
# - `source_type` : The source of the water
# - `source_class` : The source of the water
# - `waterpoint_type` : The kind of waterpoint
# - `waterpoint_type_group` : The kind of waterpoint
# 
# ### Labels
# 
# There are three possible values:
# 
# - `functional` : the waterpoint is operational and there are no repairs needed
# - `functional needs repair` : the waterpoint is operational, but needs repairs
# - `non functional` : the waterpoint is not operational
# 
# --- 
# 
# ## Generate a submission
# 
# Your code to generate a submission file may look like this:
# 
# ```python
# # estimator is your model or pipeline, which you've fit on X_train
# 
# # X_test is your pandas dataframe or numpy array, 
# # with the same number of rows, in the same order, as test_features.csv, 
# # and the same number of columns, in the same order, as X_train
# 
# y_pred = estimator.predict(X_test)
# 
# 
# # Makes a dataframe with two columns, id and status_group, 
# # and writes to a csv file, without the index
# 
# sample_submission = pd.read_csv('sample_submission.csv')
# submission = sample_submission.copy()
# submission['status_group'] = y_pred
# submission.to_csv('your-submission-filename.csv', index=False)
# ```
# 
# If you're working locally, the csv file is saved in the same directory as your notebook.
# 
# If you're using Google Colab, you can use this code to download your submission csv file.
# 
# ```python
# from google.colab import files
# files.download('your-submission-filename.csv')
# ```
# 
# ---

#%%
import os, sys
in_colab = 'google.colab' in sys.modules

# If you're in Colab...
if in_colab:
    # Pull files from Github repo
    os.chdir('/content')
    get_ipython().system('git init .')
    get_ipython().system('git remote add origin https://github.com/LambdaSchool/DS-Unit-2-Regression-Classification.git')
    get_ipython().system('git pull origin master')
    
    # Install required python packages
    get_ipython().system('pip install -r requirements.txt')
    
    # Change into directory for module
    os.chdir('module4')


#%%
# Ignore this Numpy warning when using Plotly Express:
# FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning, module='numpy')


#%%
# Read the Tanzania Waterpumps data
# train_features.csv : the training set features
# train_labels.csv : the training set labels
# test_features.csv : the test set features
# sample_submission.csv : a sample submission file in the correct format
    
import pandas as pd

train_features = pd.read_csv('../data/waterpumps/train_features.csv')
train_labels = pd.read_csv('../data/waterpumps/train_labels.csv')
test_features = pd.read_csv('../data/waterpumps/test_features.csv')
sample_submission = pd.read_csv('../data/waterpumps/sample_submission.csv')

assert train_features.shape == (59400, 40)
assert train_labels.shape == (59400, 2)
assert test_features.shape == (14358, 40)
assert sample_submission.shape == (14358, 2)


#%%
train_features.shape

#%%
train_labels.shape

#%%
test_features.shape
#%%
sample_submission.shape

#%%
train_features.head(45)
#%%
train_labels.head(45)

#%%
import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#%%
train_features['id'].isin(train_labels['id']).value_counts()
#%%
train = pd.merge(train_labels, train_features, how='right', on=['id'])
#%%
train.head()
#%%
train.status_group.replace({'functional':0, 'functional needs repair':1, 'non functional':2}, inplace=True)
#%%
training_day, training_val = train_test_split(train, random_state=42)
training_day.shape, training_val.shape
#%%
# BASELINE FOR CLASS STATUS_GROUP
target = "status_group"
y_train = training_day[target]
y_train.value_counts(normalize = True)
#%%
y_train.mode()[0]
# the majority would thus be functional, though a worry would be that many are non functional
#%%
major_class = y_train.mode()[0]
y_pred1 = [major_class] * len(y_train)

#%%
sum(abs(y_pred1 - y_train))/ len(y_train)

#%%
# baseline prdictions validate what we observe from normalized values
accuracy_score(y_train, y_pred1)

#%%
# lets test the actual validation set
y_val = training_val[target]
y_pred2 = [major_class] * len(y_val)
accuracy_score(y_val, y_pred2)
#%%
# i should take a look at some of the features to see what i include in the feature, asking myself what
# can impact the functioning of water


train.head()

# i like subvillage, region, installer, population, source_class, waterpoint_type_group, quantity_group, gps_height, region, scheme_management, management_group, payment
#%%
target = "status_group"
features =['subvillage', 'region', 'installer', 'population', 'source_class', 'waterpoint_type_group', 'quantity_group', 'gps_height', 'region', 'scheme_management', 'management_group', 'payment']


X_train = training_day[features]
y_train = training_day[target]
X_val = training_val[features]
y_val = training_val[target]

#%%
encoder = ce.OneHotEncoder(use_cat_names=True)
X_train_encoded = encoder.fit_transform(X_train)
X_val_encoded = encoder.transform(X_val)

#%%
imputer = SimpleImputer()
X_train_imputed = imputer.fit_transform(X_train_encoded)
X_val_imputed = imputer.transform(X_val_encoded)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_val_scaled = scaler.transform(X_val_imputed)

#%%
