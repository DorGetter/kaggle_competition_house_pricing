#!/usr/bin/env python
# coding: utf-8

# # Creating Dataset for training
# ## preprocess the data

# In[21]:


import pandas as pd


# In[22]:


def read_data():
    train_df = pd.read_csv('./data/train.csv')
    test_df = pd.read_csv('./data/test.csv')
    print("train df: {}, test_df: {}".format(train_df.shape, test_df.shape))
    dataset = pd.concat([train_df, test_df], axis=0)
#     print(dataset.iloc[1460])
    return dataset
dataset = read_data()
dataset.isna().sum()


# In[23]:


dataset.shape
dataset.head(2)


# In[24]:


def fill_NaNs(df):
    df.drop('Id', axis=1, inplace=True) # removing Id feature (will not give any info on the price just fifo).
    df['PoolQC'] = df['PoolQC'].fillna('NA') # No pool
    df['MiscFeature'] = df['MiscFeature'].fillna('NA') # no special element in the house.
    df['Alley'] = df['Alley'].fillna('NA') # not access to alley
    df['Fence'] = df['Fence'].fillna('NA') # no fence 
    # same thing we will do to FireplaceQu, LotFrontage
    df['FireplaceQu'] = df['FireplaceQu'].fillna('NA') # no fireplace in the house. 
    df['LotFrontage'] = df['LotFrontage'].fillna(0.) # there is no front area. 
    # and for the Garage missing houses and the Basement missing houses. 
    df.fillna({'GarageType':'NA', 'GarageFinish':'NA', 'GarageQual':'NA', 'GarageCond':'NA', 'GarageYrBlt':.0}, inplace=True)
    df.fillna({'BsmtExposure':'NA', 'BsmtQual':'NA', 'BsmtFinType2':'NA', 'BsmtCond':'NA', 'BsmtFinType1': 'NA'}, inplace=True)
    df['MasVnrType'] = df['MasVnrType'].fillna('NA') # No Masonry veneer type
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0.) # No Masonry area. 
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0]) # we will replace the NaNs with the median=SBrkr
    return df


# In[25]:


dataset = fill_NaNs(dataset)


# In[26]:


def create_dummies(df):
    df = pd.get_dummies(df, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=True, dtype=None)
    return df


# In[27]:


dataset = create_dummies(dataset)
dataset.columns.tolist()


# In[28]:


train_df, kaggle_test_df = dataset[~dataset['SalePrice'].isna()] ,  dataset[dataset['SalePrice'].isna()] # split back. 


# In[29]:


print(train_df.shape, kaggle_test_df.shape)


# # Modeling

# In[30]:


from sklearn.model_selection import train_test_split
X = train_df.drop('SalePrice', axis=1)
y = train_df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)


# In[31]:


print(X_train.shape, X_test.shape,y_train.shape,y_test.shape)


# #### import dependencies

# In[32]:


from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# In[33]:


piplines = {
    'rf':make_pipeline(RandomForestRegressor(random_state=1234)),
    'gb':make_pipeline(GradientBoostingRegressor(random_state=1234)),
    'ridge':make_pipeline(Ridge(random_state=1234)),
    'lasso':make_pipeline(Lasso(random_state=1234)),
    'enet':make_pipeline(ElasticNet(random_state=1234)),
}


# ###### create hyperparameter grid.

# In[34]:


RandomForestRegressor().get_params()


# In[35]:


hypergrid = {
    'rf':{
        'randomforestregressor__min_samples_split':[2,3,4,5,6],
        'randomforestregressor__min_samples_leaf':[1,2,3,4,5,6]
    },
    
    'gb':{
        'gradientboostingregressor__alpha':[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.99]
    },
    'ridge':{
        'ridge__alpha':[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.99]
    },
    'lasso':{
        'lasso__alpha':[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.99]
    },
    'enet':{
        'elasticnet__alpha':[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.99]
    }
}


# In[36]:


from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV


# In[37]:


fit_models = {}
for algo, pipline in piplines.items():
    model = GridSearchCV(pipline, hypergrid[algo], cv=10, n_jobs=-1)
    try:
        print("starting training for {}".format(algo))
        model.fit(X_train, y_train)
        fit_models[algo] = model
        print("{} trained succefully!".format(algo))
    except NotFittedError as e:
        print(repr(e))


# # Evaluation 

# In[61]:


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np


# In[39]:


results = {
    'rf': None,
    'gb':None,
    'ridge':None,
    'lasso':None,
    'enet':None
}


# In[68]:


import pickle
for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    results[algo] = np.round(yhat,2)
    print('{} scores - R2: {} MAE: {} RMSE: {}'.format(algo, r2_score(y_test, yhat), mean_absolute_error(y_test, yhat), mean_squared_error(y_test,yhat)))


# it seems that our winner is the grdientboosting regressor!

# In[41]:


df_res = pd.DataFrame(results)


# In[42]:


df_res.shape
df_res['GT'] = list(y_test)
df_res['diff_rf'] = df_res['GT'] - df_res['rf']
df_res['diff_gb'] = df_res['GT'] - df_res['gb']
df_res['diff_ridge'] = df_res['GT'] - df_res['ridge']
df_res['diff_lasso'] = df_res['GT'] - df_res['lasso']
df_res['diff_enet'] = df_res['GT'] - df_res['enet']


# ## Analysing results

# In[43]:


df_rf_res = df_res[['diff_rf', 'rf', 'GT']]
df_rf_res.head(5)


# In[44]:


df_gb_res = df_res[['diff_gb', 'gb', 'GT']]
df_gb_res.head(5)


# In[45]:


df_ridge_res = df_res[['diff_ridge', 'ridge', 'GT']]
df_ridge_res.head(5)


# In[46]:


df_lasso_res = df_res[['diff_lasso', 'lasso', 'GT']]
df_lasso_res.head(5)


# In[47]:


df_enet_res = df_res[['diff_enet', 'enet', 'GT']]
df_enet_res.head(5)


# ##### Check time to predict:

# In[48]:


import time


# In[49]:


dict_avg = {
    'rf': [],
    'gb':[],
    'ridge':[],
    'lasso':[],
    'enet':[]
}


# In[50]:


def time_decorator(orignal_function):
    def time_wrapper(*args, **kwargs):
        st = time.time()
        orignal_function(*args, **kwargs)
        dict_avg[args[0]]= time.time() - st
        return  
    return time_wrapper


# In[51]:


@time_decorator
def run_performance_test(algo, model, x_test):
    yhat = model.predict(X_test)


# In[59]:


for algo, model in fit_models.items():
    run_performance_test(algo, model, X_test)


# In[60]:


for algo, val in dict_avg.items():
    print(algo, val)
# dict_avg['rf']/len(X_test) * 100


# In[ ]:




