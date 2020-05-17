#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


#loading training and test data
train_data=pd.read_csv(r'D:\DATA SCIENCE DOCS\Python docs\risk_analytics_train.csv',header=0)
test_data=pd.read_csv(r'D:\DATA SCIENCE DOCS\Python docs\risk_analytics_test.csv', header=0)


# **Preprocessing the training dataset**

# In[3]:


print(train_data.shape)

train_data.head()


# In[4]:


#finding the missing values
print(train_data.isnull().sum())
#print(train_data.shape)


# In[5]:


#imputing categorical missing data with mode value

colname1=["Gender","Married","Dependents","Self_Employed", "Loan_Amount_Term"]

for x in colname1:
    train_data[x].fillna(train_data[x].mode()[0],inplace=True)


# In[6]:


print(train_data.isnull().sum())


# In[7]:


#imputing numerical missing data with mean value
train_data["LoanAmount"].fillna(train_data["LoanAmount"].mean(),inplace=True)
print(train_data.isnull().sum())


# In[8]:


#imputing values for credit_history column differently
train_data['Credit_History'].fillna(value=0, inplace=True)
#train_data['Credit_History']=train_data['Credit_History'].fillna(value=0)
print(train_data.isnull().sum())


# In[9]:


train_data.Credit_History.mode()


# In[10]:


#transforming categorical data to numerical
from sklearn import preprocessing
colname=['Gender','Married','Education','Self_Employed','Property_Area',
         'Loan_Status']

#le={}

le=preprocessing.LabelEncoder()

for x in colname:
     train_data[x]=le.fit_transform(train_data[x])
        
#converted Loan status as Y-->1 and N-->0


# In[11]:


train_data.head()


# **Preprocessing the testing dataset**

# In[12]:


test_data.head()


# In[13]:


#finding the missing values
print(test_data.isnull().sum())
print(test_data.shape)


# In[14]:


#imputing missing data with mode value
colname1=["Gender","Dependents","Self_Employed", "Loan_Amount_Term"]


for x in colname1:
    test_data[x].fillna(test_data[x].mode()[0],inplace=True)


# In[15]:


print(test_data.isnull().sum())


# In[16]:


#imputing numerical missing data with mean value

test_data["LoanAmount"].fillna(test_data["LoanAmount"].mean(),inplace=True)
print(test_data.isnull().sum())


# In[17]:


#imputing values for credit_history column differently
test_data['Credit_History'].fillna(value=0, inplace=True)
print(test_data.isnull().sum())


# In[18]:


#transforming categorical data to numerical

from sklearn import preprocessing

colname=['Gender','Married','Education','Self_Employed','Property_Area']

#le={}

le=preprocessing.LabelEncoder()

for x in colname:
     test_data[x]=le.fit_transform(test_data[x])


# In[19]:


test_data.head()


# **Creating training and testing datasets and running the model**

# In[23]:


X_train=train_data.values[:,1:-1]
Y_train=train_data.values[:,-1]
Y_train=Y_train.astype(int)


# In[24]:


#test_data.head()
X_test=test_data.values[:,1:]


# In[25]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train) #fit() should be used on training data only
#transform() should be used on both training and testing data
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)


# In[26]:


from sklearn import svm
svc_model=svm.SVC(kernel='rbf',C=1.0,gamma=0.1)
#svc- support vector classifier
#svr- support vector regression
#from sklearn.linear_model import LogisticRegression
#svc_model=LogisticRegression()
svc_model.fit(X_train, Y_train)
Y_pred=svc_model.predict(X_test)
print(list(Y_pred))


# In[27]:


Y_pred_col=list(Y_pred)
#print(Y_pred_col)


# In[28]:


test_data=pd.read_csv(r'D:\DATA SCIENCE DOCS\Python docs\risk_analytics_test.csv',header=0)
test_data["Y_predictions"]=Y_pred_col
test_data.head()


# In[ ]:
test_data.to_csv('test_data.csv')


# In[35]:
#Using cross validation
from sklearn.linear_model import LogisticRegression
classifier=(LogisticRegression()) #output is 77.2% 
#here logistic reg gave best accuracy than svm
#svm gave accuracy 75% without any tuning and using default value
#but after tuning it gave 77.02% which is less than log regression accuracy
#hence log regression outperforms for binary values
#we cant keep on increasing C and gamma value drastically bcz it may lead to misclassification

#classifier=svm.SVC(kernel="rbf",C=1.0,gamma=0.1) #output is 75.89%   #default value of gamma and C

#classifier=svm.SVC(kernel="rbf",C=10.0,gamma=0.001) # output is 77.03

#classifier=svm.SVC(kernel="linear",C=10.0,gamma=0.001) # output is 77.03

#classifier=svm.SVC(kernel="poly",C=10.0,gamma=0.001) # output is 68.73

#performing kfold_cross_validation
from sklearn.model_selection import KFold
kfold_cv=KFold(n_splits=10)
print(kfold_cv)

from sklearn.model_selection import cross_val_score
#running the model using scoring metric as accuracy
kfold_cv_result=cross_val_score(estimator=classifier,X=X_train, y=Y_train, cv=kfold_cv)
print(kfold_cv_result)
#finding the mean
print(kfold_cv_result.mean())

"""
for train_value, test_value in kfold_cv.split(X_train):
    classifier.fit(X_train[train_value], Y_train[train_value]).predict(X_train[test_value])

    
Y_pred=classifier.predict(X_test)
#print(list(zip(Y_test,Y_pred)))
"""


# In[ ]:





# In[ ]:


for x in range(0,len(Y_pred_col)):

    if Y_pred_col[x]==0:
        Y_pred_col[x]= "N"
    else:
        Y_pred_col[x]="Y"
    
print(Y_pred_col)

