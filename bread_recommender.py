#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
bread=pd.read_csv('bread.csv')
bread


# In[2]:


x = bread.drop(columns=['Bread'])
x


# In[3]:


y = bread['Bread']
y


# In[4]:


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(x.values,y)
prediction = model.predict([[35,0],[44,1]])
prediction


# In[5]:


import joblib
joblib.dump(model,'bread-recommender.joblib')


# In[ ]:




