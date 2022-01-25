#!/usr/bin/env python
# coding: utf-8

# # Task Assigned use Knn Algorithm on Nba 2013 Dataset
# # (Parth Madan)

# In[1]:


#Import libraries
import pandas as pd 
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics


# In[2]:


data = pd.read_csv(r'C:\Users\Parth Madan\Downloads\nba_2013.csv')
data.head()


# # Data Pre Processing

# In[3]:


data.isnull().any()


# # Fill Null Values with its mean Value

# In[4]:


data["fg."].fillna(data["fg."].mean(),inplace=True)
data["x2p."].fillna(data["x2p."].mean(),inplace=True)
data["efg."].fillna(data["efg."].mean(),inplace=True)
data["x3p."].fillna(data["x3p."].mean(),inplace=True)
data["ft."].fillna(data["ft."].mean(),inplace=True)


# # Select Valid Numeric Columns from respective dataset

# In[5]:


distance_columns = ['age', 'g', 'gs', 'mp', 'fg', 'fga', 'fg.', 'x3p', 'x3pa', 'x3p.', 'x2p', 'x2pa', 'x2p.', 'efg.', 'ft', 'fta', 'ft.', 'orb', 'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf', 'pts']
data_numeric = data[distance_columns]


# In[6]:


data_numeric


# # Apply Normalization

# In[7]:


data_normalized = data_numeric.apply(lambda x: (x - x.min()) / (x.max() - x.min()))


# In[8]:


#categorical Columns
data_category = data[['player', 'bref_team_id', 'season']]


# # Train Test Split

# In[9]:


data = pd.concat([data_category, data_normalized], axis=1)

from sklearn.model_selection import train_test_split

# The columns that we will be making predictions with.
x_columns = data[['age', 'g', 'gs', 'mp', 'fg', 'fga', 'fg.', 'x3p', 'x3pa', 'x3p.', 'x2p', 'x2pa', 'x2p.', 'efg.', 'ft', 'fta', 'ft.', 'orb', 'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf']]

# The column that we want to predict.
y_column = data["pts"]

x_train, x_test, y_train, y_test = train_test_split(x_columns, y_column, test_size=0.3, random_state=0) #70%train and 30% Test


# # Create Knn Model

# In[10]:


for k in range(10):
    k_value = k + 1
    knn = KNeighborsRegressor(n_neighbors = k_value)
    knn.fit(x_train, y_train) 
    y_pred = knn.predict(x_test)
    print ("Regression score is:",format(metrics.r2_score(y_test, y_pred),'.4f'), "for k_value:", k_value)


# In[11]:


#As we got Highest value at k=8
knn = KNeighborsRegressor(n_neighbors = 8)
knn.fit(x_train, y_train) 
y_pred = knn.predict(x_test)
print ("Mean Squared Error is:", format(metrics.mean_squared_error(y_test, y_pred), '.7f'))
print ("Regression score is:", format(metrics.r2_score(y_test, y_pred),'.4f'))


# In[12]:


Predicted_value = pd.DataFrame({'Actual Points': y_test.tolist(), 'Predicted Points': y_pred.tolist()})

Predicted_value


# # END

# In[ ]:




