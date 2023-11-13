#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer


# In[2]:


df= pd.read_csv('titanic_dataset.csv')


# In[3]:


df


# In[4]:


df.shape


# In[5]:


df.head(5)


# In[6]:


df.tail()


# In[7]:


df.describe()


# In[8]:


df.info()


# In[9]:


df.notnull()


# In[10]:


df.dropna()


# In[11]:


df.isnull().sum()


# In[12]:


df= df.fillna(0)


# In[13]:


df=df.drop('Cabin',axis=1)


# In[14]:


df['Age'].fillna(df['Age'].mean(),inplace=True)


# In[15]:


df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)


# In[16]:


df.isnull().sum()


# In[17]:


df.describe()


# In[18]:


df['Age'].value_counts()


# In[19]:


df['Survived'].value_counts()


# In[20]:


sns.barplot(x='Pclass',y='Age' ,data=df )
plt.show()


# In[21]:


sns.histplot(x='Survived',data=df)
plt.show()


# In[22]:


sns.countplot(x=df["Survived"])


# In[23]:


sns.countplot(x='Sex',hue= 'Survived', data=df)


# In[24]:


sns.countplot(x="Age",data=df)


# In[25]:


sns.pairplot(df, hue='Age', height=2)


# In[26]:


df['Sex'].value_counts()


# In[27]:


df['Embarked'].value_counts()


# In[28]:


df= df.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}})


# In[29]:


df.head()


# In[30]:


X=df.drop(columns=['PassengerId','Name','Ticket','Survived'],axis=1)
Y=df['Survived']


# In[31]:


X


# In[32]:


Y


# In[33]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)


# In[34]:


x_tr,x_test,y_tr,y_test=train_test_split(X,Y,test_size=0.2,random_state=2)


# In[35]:


print(X.shape, X_train.shape, X_test.shape)


# In[36]:


model = LogisticRegression()


# In[37]:


model.fit(X_train, Y_train)


# In[38]:


LogisticRegression()


# In[39]:


X_train_prediction = model.predict(X_train)


# In[40]:


print(X_train_prediction)


# In[41]:


training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)


# In[42]:


# accuracy on test data
X_test_prediction = model.predict(X_test)


# In[43]:


print(X_test_prediction)


# In[44]:


test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)


# In[ ]:




