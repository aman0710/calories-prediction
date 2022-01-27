#!/usr/bin/env python
# coding: utf-8

# ### Importing the dependencies

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics


# ### Data Collection and Pre-Processing

# In[2]:


# loading the data from csv file to pandas dataframe

calories = pd.read_csv('calories.csv')


# In[3]:


# print the first 5 rows of the dataframe

calories.head()


# In[4]:


exercise = pd.read_csv('exercise.csv')


# In[5]:


exercise.head()


# ### Combining the two dataframes

# In[6]:


calories_data = pd.concat([exercise, calories['Calories']], axis = 1)


# In[7]:


calories_data.head()


# In[8]:


# checking the number of rows and columns

calories_data.shape


# In[9]:


# getting some information about the data

calories_data.info()


# In[10]:


# checking for missing values

calories_data.isnull().sum()


# ### Data Analysis

# In[11]:


# getting some statistics about the data

calories_data.describe()


# ### Data Visualization

# In[12]:


sns.set()


# In[13]:


# plotting the gender column in count plot

sns.countplot(x = calories_data['Gender'])


# In[14]:


# finding the distribution of "Age" column

sns.displot(calories_data['Age'])


# In[15]:


# finding the distribution of "Height" column

sns.displot(calories_data['Height'])


# In[16]:


# finding the distribution of "Weight" column

sns.displot(calories_data['Weight'])


# ### Finding the correlation in the dataset

# 1. Positive Correlation
# 2. Negative Correlation

# In[17]:


correlation = calories_data.corr()


# In[18]:


# constructing a heatmap to understand the correlation

plt.figure(figsize = (10, 10))
sns.heatmap(correlation, cbar = True, square = True, fmt = '.1f', annot = True, annot_kws = {'size': 8}, cmap = 'Blues')


# ### Converting the text data to numerical values

# Male: 0; Female: 1

# In[19]:


calories_data.replace({'Gender': {'male': 0, 'female': 1}}, inplace = True)


# In[20]:


calories_data.head()


# ### Separating the features and labels

# In[21]:


X = calories_data.drop(columns = ['User_ID', 'Calories'], axis = 1)
Y = calories_data['Calories']


# In[22]:


X


# In[23]:


Y


# ### Splitting the data into training data and test data

# In[24]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)


# In[25]:


print(X.shape, X_train.shape, X_test.shape)


# ### Model Training

# XGBoost Regressor

# In[26]:


model = XGBRegressor(objective = 'reg:squarederror')


# In[27]:


# training the model with X_train

model.fit(X_train, Y_train)


# ### Evaluating the model

# #### Prediction on training data

# In[28]:


prediction_train = model.predict(X_train)


# In[29]:


print(prediction_train)


# Mean Absolute Error

# In[30]:


mae_train = metrics.mean_absolute_error(Y_train, prediction_train)


# In[31]:


print('Mean Absolute Error : ', mae_train)


# R-squared value

# In[32]:


r2_train = metrics.r2_score(Y_train, prediction_train)


# In[33]:


print('R-squared value : ', r2_train)


# Visualize the actual calories and predicted calories

# In[34]:


plt.scatter(Y_train, prediction_train)
plt.xlabel('Actual Calories')
plt.ylabel('Predicted Calories')
plt.title('Actual Calories vs. Predicted Calories')
plt.show()


# #### Prediction on test data

# In[35]:


prediction_test = model.predict(X_test)


# In[36]:


print(prediction_test)


# Mean Absolute Error

# In[37]:


mae_test = metrics.mean_absolute_error(Y_test, prediction_test)


# In[38]:


print('Mean Absolute Error : ', mae_test)


# R-squared value

# In[39]:


r2_test = metrics.r2_score(Y_test, prediction_test)


# In[40]:


print('R-squared value : ', r2_test)


# Visualize the actual calories and predicted calories

# In[41]:


plt.scatter(Y_test, prediction_test)
plt.xlabel('Actual Calories')
plt.ylabel('Predicted Calories')
plt.title('Actual Calories vs. Predicted Calories')
plt.show()


# ### Building a Predictive System

# In[42]:


# making a pandas dataframe containing one row and predicting the output. That one row is X_test.iloc[0]

input_data = pd.DataFrame([[1, 41, 172.0, 74.0, 24.0, 98.0, 40.8]], 
                          columns = ['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])


prediction = model.predict(input_data)


# In[43]:


print('Calories burnt : ', prediction[0])


# ### Saving the Model

# Save Model

# In[44]:


with open('calories_predictor.pkl', 'wb') as f:
    pickle.dump(model, f)


# Load Model

# In[45]:


with open('calories_predictor.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

