# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'module3'))
	print(os.getcwd())
except:
	pass
#%%
from IPython import get_ipython

#%% [markdown]
# <a href="https://colab.research.google.com/github/qweliant/DS-Unit-2-Regression-Classification/blob/master/assignment_regression_classification_3.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#%% [markdown]
# Lambda School Data Science, Unit 2: Predictive Modeling
# 
# # Regression & Classification, Module 3
# 
# ## Assignment
# 
# We're going back to our other **New York City** real estate dataset. Instead of predicting apartment rents, you'll predict property sales prices.
# 
# But not just for condos in Tribeca...
# 
# Instead, predict property sales prices for **One Family Dwellings** (`BUILDING_CLASS_CATEGORY` == `'01 ONE FAMILY DWELLINGS'`). 
# 
# Use a subset of the data where the **sale price was more than \\$100 thousand and less than $2 million.** 
# 
# The [NYC Department of Finance](https://www1.nyc.gov/site/finance/taxes/property-rolling-sales-data.page) has a glossary of property sales terms and NYC Building Class Code Descriptions. The data comes from the [NYC OpenData](https://data.cityofnewyork.us/browse?q=NYC%20calendar%20sales) portal.
# 
# - [ ] Do train/test split. Use data from January — March 2019 to train. Use data from April 2019 to test.
# - [ ] Do one-hot encoding of categorical features.
# - [ ] Do feature selection with `SelectKBest`.
# - [ ] Do [feature scaling](https://scikit-learn.org/stable/modules/preprocessing.html).
# - [ ] Fit a ridge regression model with multiple features.
# - [ ] Get mean absolute error for the test set.
# - [ ] As always, commit your notebook to your fork of the GitHub repo.
# 
# 
# ## Stretch Goals
# - [ ] Add your own stretch goal(s) !
# - [ ] Instead of `RidgeRegression`, try `LinearRegression`. Depending on how many features you select, your errors will probably blow up! 💥
# - [ ] Instead of `RidgeRegression`, try [`RidgeCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html).
# - [ ] Learn more about feature selection:
#     - ["Permutation importance"](https://www.kaggle.com/dansbecker/permutation-importance)
#     - [scikit-learn's User Guide for Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)
#     - [mlxtend](http://rasbt.github.io/mlxtend/) library
#     - scikit-learn-contrib libraries: [boruta_py](https://github.com/scikit-learn-contrib/boruta_py) & [stability-selection](https://github.com/scikit-learn-contrib/stability-selection)
#     - [_Feature Engineering and Selection_](http://www.feat.engineering/) by Kuhn & Johnson.
# - [ ] Try [statsmodels](https://www.statsmodels.org/stable/index.html) if you’re interested in more inferential statistical approach to linear regression and feature selection, looking at p values and 95% confidence intervals for the coefficients.
# - [ ] Read [_An Introduction to Statistical Learning_](http://faculty.marshall.usc.edu/gareth-james/ISL/ISLR%20Seventh%20Printing.pdf), Chapters 1-3, for more math & theory, but in an accessible, readable way.
# - [ ] Try [scikit-learn pipelines](https://scikit-learn.org/stable/modules/compose.html).
#%% [markdown]
# # New Section

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
    os.chdir('module3')


#%%
# Ignore this Numpy warning when using Plotly Express:
# FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning, module='numpy')


#%%
import pandas as pd
import pandas_profiling

# Read New York City property sales data
df = pd.read_csv('../data/condos/NYC_Citywide_Rolling_Calendar_Sales.csv')

# Change column names: replace spaces with underscores
df.columns = [col.replace(' ', '_') for col in df]

# SALE_PRICE was read as strings.
# Remove symbols, convert to integer
df['SALE_PRICE'] = (
    df['SALE_PRICE']
    .str.replace('$','')
    .str.replace('-','')
    .str.replace(',','')
    .astype(int)
)


#%%
# BOROUGH is a numeric column, but arguably should be a categorical feature,
# so convert it from a number to a string
df['BOROUGH'] = df['BOROUGH'].astype(str)


#%%
# Reduce cardinality for NEIGHBORHOOD feature

# Get a list of the top 10 neighborhoods
top10 = df['NEIGHBORHOOD'].value_counts()[:10].index

# At locations where the neighborhood is NOT in the top 10, 
# replace the neighborhood with 'OTHER'
df.loc[~df['NEIGHBORHOOD'].isin(top10), 'NEIGHBORHOOD'] = 'OTHER'
df.SALE_DATE = pd.to_datetime(df.SALE_DATE)
df2 = df


#%%
df.head()


#%%
df.describe()


#%%
df.shape


#%%

cut = ((df.SALE_PRICE > 100000) & (df.SALE_PRICE < 2000000) & (df['BUILDING_CLASS_CATEGORY'] == '01 ONE FAMILY DWELLINGS'))


#%%
df2 = df2[cut] 
print(df2.shape)
df2.head()


#%%



#%%



#%%
df2.LAND_SQUARE_FEET = [ float(i.replace(',','')) for i in df2.LAND_SQUARE_FEET]
df2.dtypes


#%%
print(df2.shape)
df2.head()


#%%
# drop becasue it goes hand in hand with BUILDING_CLASS_CATEGORY, EASE-MENT becasue its mostly NaNs
df2.drop(columns=['TAX_CLASS_AT_PRESENT', 'EASE-MENT', 'APARTMENT_NUMBER'], inplace=True)


#%%
df2.head()


#%%
print(df2.BOROUGH.unique(),'\n')
print(df2.NEIGHBORHOOD.unique(), '\n')

print(df2.BUILDING_CLASS_AT_PRESENT.unique(), '\n')
print(df2.BUILDING_CLASS_AT_TIME_OF_SALE.unique(), '\n')
print(df2.ZIP_CODE.unique() , '\n')


#%%
df2.replace({'A3':1, 'A7':2, 'A9':3, 'A4':4, 'A1':5, 'S1':6, 'S0':7, 'A5':8}, inplace =True)


#%%
df2.NEIGHBORHOOD.value_counts()


#%%
df2.NEIGHBORHOOD.replace({'OTHER': 1, 'UPPER EAST SIDE (59-79)': 2, 'FOREST HILLS':3,  'UPPER EAST SIDE (79-96)':4, 'UPPER WEST SIDE (79-96)':5,  'UPPER EAST SIDE (79-96)':6 , 'BEDFORD STUYVESANT':7}, inplace=True)
df2.NEIGHBORHOOD.value_counts()


#%%
df2.head(3)


#%%


#%% [markdown]
# 

#%%
cutoff = pd.to_datetime('2019-04-01')
train = df2[df2.SALE_DATE < cutoff]
test  = df2[df2.SALE_DATE >= cutoff]
train.shape, test.shape


#%%
import pandas_profiling


#%%
train.profile_report()


#%%
train.describe(exclude='number').T.sort_values(by='unique')


#%%
target = 'SALE_PRICE'
high_cardinality  = ['SALE_DATE', 'ADDRESS']
feature = train.columns.drop([target] + high_cardinality)

X_train = train[feature]
y_train = train[target]
X_test = test[feature]
y_test = test[target]


#%%
X_train.head()


#%%
import category_encoders as ce
encoder = ce.OneHotEncoder(use_cat_names=True)
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)


#%%
X_train.head()


#%%
warnings.filterwarnings(action='ignore', category=RuntimeWarning, module='sklearn')
warnings.filterwarnings(action='ignore', category=RuntimeWarning, module='scipy')


#%%
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


#%%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_test_scaled.shape, X_train_scaled


#%%
for k in range(1, len(X_train.columns)+1):
    
    print(f'{k} features')
    
    selector = SelectKBest(score_func=f_regression, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Test MAE: ${mae:,.0f} \n')


#%%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

for k in range(1, len(X_train.columns)+1):
    print(f'{k} features')
    
    selector = SelectKBest(score_func=f_regression, k=k)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    model = RidgeCV()
    model.fit(X_train_selected, y_train)
    
    y_pred = model.predict(X_test_selected)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Test MAE: ${mae:,.0f} \n')
    


#%%
# Which features were selected?
all = X_train.columns
mask = selector.get_support()
select_names = all[mask]
unselect_names = all[~mask]

print('Features selected:')
for name in select_names:
    print(name)
    
print('\n')
print('Features not selected:')
for name in unselect_names:
    print(name)


