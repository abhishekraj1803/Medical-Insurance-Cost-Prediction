#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LinearRegression


# In[2]:


df = pd.read_csv(r"C:\Users\hplap\Downloads\Telegram Desktop\insurance.csv")
df.head()


# In[4]:


df.shape


# In[7]:


df.info()


# In[9]:


df.describe()


# In[10]:


sns.set()
plt.figure(figsize = (6,6))
sns.distplot(df['age'])
plt.title("Age Distribution")
plt.show()


# In[11]:


plt.figure(figsize = (6,6))
sns.countplot(x="sex",data = df)
plt.title("Sex Distribution")
plt.show()


# In[12]:


df['sex'].value_counts()


# In[13]:


sns.distplot(df["bmi"])
plt.show()


# In[14]:


df['region'].value_counts()


# In[15]:


df.replace({'sex':{'male':0,'female':1}},inplace = True)
df.replace({'smoker':{'yes':0,'no':1}},inplace = True)
df.replace({'region':{'southeast':0,'southwest':1,'northwest':2,'northeast':3}},inplace =True)


# In[16]:


x = df.drop(columns = 'charges',axis = 1)
y = df["charges"]


# In[17]:


x


# In[18]:


y


# In[19]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)


# In[20]:


x_train.shape


# In[21]:


x_test.shape


# In[22]:


reg = LinearRegression()


# In[23]:


reg.fit(x_train,y_train)


# In[24]:


data = reg.predict(x_train)


# In[25]:


r2_train = metrics.r2_score(y_train,data)
r2_train


# In[26]:


test_data_prediction = reg.predict(x_test)


# In[27]:


metrics.r2_score(y_test,test_data_prediction)


# In[28]:


sample = (30,1,22.7,0,1,0)


# In[29]:


input_data_as_numpy_array = np.asarray(sample)


# In[30]:


input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


# In[31]:


prediction  = reg.predict(input_data_reshaped)


# In[32]:


print("The insurance cost is :" , prediction)


# In[ ]:




