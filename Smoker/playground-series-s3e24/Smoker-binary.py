#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os, glob


# In[2]:


df = pd.read_csv('./train.csv')
df = df.drop(columns=['id'])


# In[3]:


df.head()


# In[6]:




X, y = df.loc[:, df.columns != "smoking"].values , df.loc[:, df.columns == "smoking"].values





# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[7]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_scaled, X_test_scaled = sc.transform(X_train), sc.transform(X_test)


# In[8]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)


# In[9]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[ ]:


from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)

print(classification_report(y_test, y_pred))


# In[ ]:




