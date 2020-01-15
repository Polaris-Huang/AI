#!/usr/bin/env python
# coding: utf-8

# In[107]:


import numpy as np
from sklearn.tree import DecisionTreeClassifier #决策树
from sklearn import preprocessing
from sklearn.metrics import accuracy_score


# In[108]:


mnist_data=np.load('./mnist.npz')


# In[109]:


mnist_data.files


# In[110]:


train_x, test_x, train_y, test_y = mnist_data['x_train'], mnist_data['x_test'], mnist_data['y_train'], mnist_data['y_test']


# In[111]:


test_x.shape


# In[112]:


#?model.fit


# In[113]:


train_x=train_x.reshape(60000,28*28)
test_x=test_x.reshape(10000,28*28)
train_y=train_y.reshape(60000,1)
test_y=test_y.reshape(10000,1)


# In[114]:


# 创建线性 CART决策树分类器
model = DecisionTreeClassifier()
model.fit(train_x,train_y)
predict_y=model.predict(test_x)
print('CART决策树准确率: %0.4lf' %accuracy_score(predict_y,test_y))


# In[ ]:




