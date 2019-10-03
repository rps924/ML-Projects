#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# # Final Project
# ## Rushabh Shah (1893142)
# ## CSC 478: Programming Machine Learning Applications - Winter 2018
# ## Due: Tuesday, March 18, 2019
# 
# ____
# 
# #### Final Project Objective: 
# *Lending Club Loan Analysis: https://www.kaggle.com/wendykan/lending-club-loan-data
# 
# #### Data Analysis Tasks:
# 
# 1. Supervised Learning: Classifier using k-nearest of payment status
#   1. Exploratory data analysis
#   2. Pre-processing, data cleaning and transformation
#   3. Building classifier model
#   4. Evaluation of model
# 
# 

# ## 1. Load Libraries

# In[48]:


## load libraries
import sys
from numpy import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn import neighbors, tree, naive_bayes
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# ## 2. Load Data

# In[49]:


data = pd.read_csv("loan.csv", low_memory=False)


# ### a. Data (size) Reduction
# * I am reducing the size of my sample to 20% of the original because when I was attempting to analyze the whole set, the application kept crashing.

# In[50]:


# 5% of the data without replacement
data = data.sample(frac=0.05, replace=False, random_state=123)  


# ## 3. Data Exploration
# * Used both descriptive and visualisation techniques in my data exploration phase.

# In[51]:


data.shape


# In[52]:


data.head(n=5)


# In[53]:


data.columns


# The **loan_status** column is the target!

# ### a. Number of classes?

# In[54]:


pd.unique(data['loan_status'].values.ravel())


# In[55]:


print("Amount of Classes: ", len(pd.unique(data['loan_status'].values.ravel())))


# In[56]:


len(pd.unique(data['zip_code'].values.ravel())) # want to make sure this was not too unique


# In[57]:


len(pd.unique(data['url'].values.ravel())) # drop url


# In[58]:


len(pd.unique(data['last_pymnt_d'].values.ravel()))


# In[59]:


len(pd.unique(data['next_pymnt_d'].values.ravel()))


# In[60]:


for col in data.select_dtypes(include=['object']).columns:
    print ("Column {} has {} unique instances".format( col, len(data[col].unique())) )


# ### b. Does the data contain unique customers?

# In[62]:


len(pd.unique(data['member_id'].values.ravel())) == data.shape[0]


# ### c. Drop variables with high cardinality:
# * We drop the variables with high cardinality because they are of no value to our analysis

# In[63]:


data = data.drop('id', 1) #
data = data.drop('member_id', 1)#
data = data.drop('url', 1)#
data = data.drop('purpose', 1)
data = data.drop('title', 1)#
data = data.drop('zip_code', 1)#
data = data.drop('emp_title', 1)#
data = data.drop('earliest_cr_line', 1)#
data = data.drop('term', 1)
data = data.drop('sub_grade', 1) #
data = data.drop('last_pymnt_d', 1)#
data = data.drop('next_pymnt_d', 1)#
data = data.drop('last_credit_pull_d', 1)
data = data.drop('issue_d', 1) ##
data = data.drop('desc', 1)##
data = data.drop('addr_state', 1)##


# In[65]:


data.shape


# In[66]:


# yay this is better
for col in data.select_dtypes(include=['object']).columns:
    print ("Column {} has {} unique instances".format( col, len(data[col].unique())) )


# ### d. Exploratory Analysis: Describing the loan amount distribution
# * The distribution of the loan amounts produced a bell-curve, showing a fairly normal distribution, with the median loan amount sitting right under $15,000.

# In[70]:


data['loan_amnt'].plot(kind="hist", bins=10)


# In[71]:


data['grade'].value_counts().plot(kind='bar')


# In[72]:


data['emp_length'].value_counts().plot(kind='bar')


# ### e. Exploratory Analysis: Describing the target class's distrbution
# * The distribution of the target class produces a poisson distribution, with a vast majority of the loans in this dataset are in a current or fully-paid off state.

# In[16]:


data['loan_status'].value_counts().plot(kind='bar')


# ### f. How many columns contain values of a numeric domain?
# 

# In[76]:


data._get_numeric_data().columns


# In[77]:


"There are {} numeric columns in the data set".format(len(data._get_numeric_data().columns) )   


# ### g. How many columns contain values of a character domain?

# In[78]:


data.select_dtypes(include=['object']).columns


# In[79]:


"There are {} Character columns in the data set (minus the target)".format(len(data.select_dtypes(include=['object']).columns) -1) 


# ## 4. Data Pre-processing:

# ### a. Removing target from dataset:

# In[80]:


X = data.drop("loan_status", axis=1, inplace = False)
y = data.loan_status


# In[81]:


y.head()


# ### b. Transforming Data into Matrix Model w/one-hot encoding:
# * As part of this transformation, I will isolate all character class variables

# In[83]:


def model_matrix(df , columns):
    dummified_cols = pd.get_dummies(df[columns])
    df = df.drop(columns, axis = 1, inplace=False)
    df_new = df.join(dummified_cols)
    return df_new

X = model_matrix(X, ['grade', 'emp_length', 'home_ownership', 'verification_status',
                    'pymnt_plan', 'initial_list_status', 'application_type', 'verification_status_joint'])

# 'issue_d' 'desc' 'addr_state'


# In[84]:


X.head()


# In[85]:


X.shape


# ### c. Scaling Continuous Variables (using min-max):

# In[86]:


# impute rows with NaN with a 0 for now
X2 = X.fillna(value = 0)
X2.head()


# In[87]:


from sklearn.preprocessing import MinMaxScaler

Scaler = MinMaxScaler()

X2[['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate',
       'installment', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths',
       'mths_since_last_delinq', 'mths_since_last_record', 'open_acc',
       'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'out_prncp',
       'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp',
       'total_rec_int', 'total_rec_late_fee', 'recoveries',
       'collection_recovery_fee', 'last_pymnt_amnt',
       'collections_12_mths_ex_med', 'mths_since_last_major_derog',
       'policy_code', 'annual_inc_joint', 'dti_joint', 'acc_now_delinq',
       'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m', 'open_il_6m',
       'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il',
       'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util',
       'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m']] = Scaler.fit_transform(X2[['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate',
       'installment', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths',
       'mths_since_last_delinq', 'mths_since_last_record', 'open_acc',
       'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'out_prncp',
       'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp',
       'total_rec_int', 'total_rec_late_fee', 'recoveries',
       'collection_recovery_fee', 'last_pymnt_amnt',
       'collections_12_mths_ex_med', 'mths_since_last_major_derog',
       'policy_code', 'annual_inc_joint', 'dti_joint', 'acc_now_delinq',
       'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m', 'open_il_6m',
       'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il',
       'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util',
       'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m']])


# In[88]:


X2.head()


# ### d. Generating a training and test set:

# In[ ]:





# In[89]:


x_train, x_test, y_train, y_test = train_test_split(X2, y, test_size=.3, random_state=123)


# In[90]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# ## 5. K-Nearest Neighbor:
# * Play around w/different values for neighbors

# In[91]:


# start out with the number of classes for neighbors
data_knn = KNeighborsClassifier(n_neighbors = 10, metric='euclidean')
data_knn


# In[92]:


data_knn.fit(x_train, y_train)


# ### a. Test the KNN model on the test set we created earlier:

# In[93]:


data_knn.predict(x_test)


# ### b. Evaluate fit of classifier model on both training and test sets using r-square values:

# In[94]:


# R-square from training and test data
rsquared_train = data_knn.score(x_train, y_train)
rsquared_test = data_knn.score(x_test, y_test)
print ('Training data R-squared:')
print(rsquared_train)
print ('Test data R-squared:')
print(rsquared_test)


# ### c. Generating the Confusion Matrix:

# In[96]:


# confusion matrix
from sklearn.metrics import confusion_matrix

knn_confusion_matrix = confusion_matrix(y_true = y_test, y_pred = data_knn.predict(x_test))
print("The Confusion matrix:\n", knn_confusion_matrix)


# In[99]:


# visualize the confusion matrix
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
plt.matshow(knn_confusion_matrix, cmap = plt.cm.Blues)
plt.title("KNN Confusion Matrix\n")
#plt.xticks([0,1], ['No', 'Yes'])
#plt.yticks([0,1], ['No', 'Yes'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
for y in range(knn_confusion_matrix.shape[0]):
    for x in range(knn_confusion_matrix.shape[1]):
        plt.text(x, y, '{}'.format(knn_confusion_matrix[y, x]),
                horizontalalignment = 'center',
                verticalalignment = 'center',)
plt.show()


# In[ ]:





# ### d. Final Classification Report:

# In[98]:


#Generate the classification report
from sklearn.metrics import classification_report
knn_classify_report = classification_report(
    y_true=y_test, y_pred=data_knn.predict(x_test))
print(knn_classify_report)


# ___
# fin.

# In[ ]:





# In[ ]:




