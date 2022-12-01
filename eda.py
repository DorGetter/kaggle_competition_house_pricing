#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno


# # EDA
# house pricing prediction
# 
# ### Reading the data

# In[ ]:


data = pd.read_csv('data/train.csv')
print("dataset types {} ".format(set(data.dtypes.tolist())))
data.head(10)


# In[ ]:


data.drop('Id', axis=1, inplace=True) # removing Id feature (will not give any info on the price just fifo).
print(len(data.columns))
data.columns # checking the features.


# In[4]:


data.head(4)


# # Handle NaNs Values:

# In[5]:


print(data.isna().mean().sort_values(ascending=False)[0:20])
msno.matrix(data)


# In[6]:


missing_fig = data.isnull().sum()
missing_fig = missing_fig[missing_fig > 0]
missing_fig.sort_values(inplace=True)
missing_fig.plot.bar()


# As we can see Fence, Alley, MiscFeature, PoolQC are hold more than 80% NaNs values so we decide to fill them with 'Na' so they will not be nones values: 
# 
# 
# forther notice: 
# - replace LotFrontage NaNs with the mean / median.
# - 

# In[7]:


def fill_NaNs(df):
    df['PoolQC'] = df['PoolQC'].fillna('NA') # No pool
    df['MiscFeature'] = df['MiscFeature'].fillna('NA') # no special element in the house.
    data['Alley'] = df['Alley'].fillna('NA') # not access to alley
    df['Fence'] = df['Fence'].fillna('NA') # no fence 
    # same thing we will do to FireplaceQu, LotFrontage
    df['FireplaceQu'] = df['FireplaceQu'].fillna('NA') # no fireplace in the house. 
    df['LotFrontage'] = df['LotFrontage'].fillna(0.) # there is no front area. 
    # and for the Garage missing houses and the Basement missing houses. 
    df.fillna({'GarageType':'NA', 'GarageFinish':'NA', 'GarageQual':'NA', 'GarageCond':'NA', 'GarageYrBlt':.0}, inplace=True)
    df.fillna({'BsmtExposure':'NA', 'BsmtQual':'NA', 'BsmtFinType2':'NA', 'BsmtCond':'NA', 'BsmtFinType1': 'NA'}, inplace=True)
    df['MasVnrType'] = df['MasVnrType'].fillna('NA') # No Masonry veneer type
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0.) # No Masonry area. 
    df['Electrical'] = df['Electrical'].fillna(data['Electrical'].mode()[0]) # we will replace the NaNs with the median=SBrkr
    return df
data = fill_NaNs(data)


# In[8]:


print(data.isna().mean().sort_values(ascending=False).head(4) ) # get the means for other missing features.
print("as we can see there is one more feature to address!")
print(f"There are { data.isnull().sum().sum()} NaNs values in the Dataset")


# # Numerical Features:

# In[9]:


numerics = ['int16', 'int32', 'int64']
print(f"There are {len(set(data._get_numeric_data().columns))} Numerical features:\n")
print(set(data._get_numeric_data().columns))
data_numerical = data._get_numeric_data().copy()
data_numerical.head(5)


# In[10]:


features_num = data_numerical.columns.tolist()
fea_cor = []
for feature in features_num:
    fea_cor.append(data[feature].corr(data['SalePrice']))
cor_num_df = pd.DataFrame(columns=['feature', 'correlation'])
cor_num_df['feature'] =features_num
cor_num_df['correlation'] =fea_cor
cor_num_df.sort_values('correlation', ascending=False)[1:].head(5)


# # Catagorical Features

# In[12]:


print("There are {} Categorical features:\n".format(len(data.select_dtypes(exclude='number').columns)))
print(data.select_dtypes(exclude='number').columns.tolist())


# ### creating dummies -> OneHotEncode the features. 

# In[ ]:


data


# In[ ]:


data = pd.get_dummies(data, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=True, dtype=None)


# In[ ]:


data.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Creating the dataset

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ##### Fixing na values in LotFrontage column - disc: Linear feet of street connected to property
# 
# there are 1201 rows with null values, are missing 259.
#  - the question is if null value is equal to 0 ?? 

# In[ ]:


data[data['LotFrontage'].isna()]
data['LotFrontage'].dropna().hist(bins=200)


# In[ ]:


mean = data['LotFrontage'].dropna().mean()
std = data['LotFrontage'].dropna().std()
median = data['LotFrontage'].dropna().median()
print(f"mean: {mean}, std: {std} ,median: {median}")
data['LotFrontage'].dropna().plot.kde()


# In[ ]:


data['LotFrontage'].fillna(median, inplace=True)
mean = data['LotFrontage'].mean()
std = data['LotFrontage'].std()
median = data['LotFrontage'].median()
print(f"mean: {mean}, std: {std} ,median: {median}")
data['LotFrontage'].plot.kde()


# ##### Fixing na values in GarageType GarageYrBlt GarageFinish GarageQual GarageCond 
# 
# 

# In[ ]:


len(data.index[data['GarageType'].isna()   &
               data['GarageYrBlt'].isna()  & 
               data['GarageFinish'].isna() &
               data['GarageQual'].isna()   &
               data['GarageCond'].isna()])


# In[ ]:


pd.set_option('display.max_columns', 76)
df_garage_null=data[data['GarageType'].isna()   &
               data['GarageYrBlt'].isna()  & 
               data['GarageFinish'].isna() &
               data['GarageQual'].isna()   &
               data['GarageCond'].isna()]
print("correlation GarageArea & df_garage_null" , len(df_garage_null[df_garage_null['GarageArea'] ==0]))
df_garage_null


# as we explored those null values are because there is no garage to the property.
# we assign them as NA -> not availble. 
# * categorical: 
#     - GarageType
#     - GarageQual	
#     - GarageCond
#     - GarageFinish
# * numerical:
#     - GarageYrBlt

# In[ ]:


data.fillna({'GarageType':'NA', 'GarageFinish':'NA', 'GarageQual':'NA', 'GarageCond':'NA', 'GarageYrBlt':.0}, inplace=True)
print(data.isna().mean().sort_values(ascending=False).head(20) )


# ##### Fixing na values in BsmtExposure BsmtFinType2 BsmtQual BsmtCond BsmtFinType1 
# 

# line 948 can be predicted as a label using all other non-null rows of features: 
# ['BsmtFinType2']
# ['BsmtQual']
# ['BsmtCond']
# ['BsmtFinType1']

# In[ ]:


data.loc[[948],'BsmtExposure'] = data['BsmtExposure'].mode()[0]
data.loc[[332],'BsmtFinType2'] =  data['BsmtFinType2'].mode()[0]


# In[ ]:


data.fillna({'BsmtExposure':'NA', 'BsmtQual':'NA', 'BsmtFinType2':'NA', 'BsmtCond':'NA', 'BsmtFinType1': 'NA'}, inplace=True)


# ##### Fixing na values in  MasVnrType      MasVnrArea      
# 

# In[ ]:


data[data['MasVnrType'].isna() & data['MasVnrArea'].isna()][['MasVnrType', 'MasVnrArea', 'OverallQual', '2ndFlrSF', 'GrLivArea', 'SalePrice']]


# In[ ]:


data['MasVnrType'].mode()


# In[ ]:





# In[ ]:





# 
