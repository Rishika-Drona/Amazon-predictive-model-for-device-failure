#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')
#import modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#from sklearn.naive_bayes import GaussianNB
#from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier
#from xgboost import XGBClassifier
#xgb=XGBClassifier()
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from xgboost import XGBClassifier

import pandas as pd
from datetime import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
#import xgboost as xgb

from sklearn.model_selection import StratifiedKFold
from imblearn.metrics import geometric_mean_score as geo
from imblearn.metrics import make_index_balanced_accuracy as iba
from sklearn.metrics import roc_curve, auc
from imblearn.metrics import geometric_mean_score, make_index_balanced_accuracy, classification_report_imbalanced
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# In[2]:


pip install pandas-profiling


# In[3]:


import pandas_profiling


# In[4]:


#---------IMPORTING THE DATA AS DATAFRAME-------------------#
# Loading the Dataset
df = pd.read_csv('C:/Users/rishi/Desktop/Amazon/predictive_maintenance.csv')

# Describing the dataset
print(df.shape)
df.head()


# In[5]:


pandas_profiling.ProfileReport(df)


# In[6]:


# Checking for the missing values (no missing values)
df.isnull().sum()


# In[7]:


# Checking the imbalanced classification problem
df.failure.value_counts()


# In[8]:


# Checking the duplicated observations (1 observation)
df.duplicated().sum()


# In[9]:


df.nunique()


# In[10]:


# Creating three more features based on date.
df.date = pd.to_datetime(df.date)

df['activedays']=df.date-df.date[0]

df['month']=df['date'].dt.month
df['week_day']=df.date.dt.weekday
df['week_day'].replace(0,7,inplace=True)
df.head()


# In[11]:


df.groupby('month').agg({'device':lambda x: x.nunique()})


# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')
df.groupby('month').agg({'device':lambda x: x.nunique()}).plot()
#(This figure shows that as time move on, the number of devices are getting less and less.)


# In[13]:


ax = sns.countplot(x="month", hue="failure", data=df)
#(This figure shows most of the devices failed in the first month.)


# In[14]:


ax = sns.countplot(x='week_day',hue='failure',data=df)
#(This figure shows that there is no device fails on Friday and Saturday. Maybe they don't work on the two days.)


# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')
df.groupby('activedays')['device'].count().plot()
#(One can see that the number of devices decreases as time goes by. And there is a big jump in the middle of activedays.) 


# In[16]:


max(df.date), min(df.date)
#(All of these data are collected between 11/02/2015 and 01/01/2015)


# In[17]:


#----Devices coming back to use-----#
df_date = df.groupby('device').agg({'date':max})

df_date.date.to_dict()

df_failure = df.loc[df.failure==1,['device','date']]


df_good = df.loc[df.failure==0,['device','date']]

df_date.shape,df_failure.shape

df['max_date']=df.device.map(df_date.date.to_dict())

df.head()

#dfa = df[~df.device.isin(df_failure.device)]

dff=df[(df.failure==1)&(df.date!=df.max_date)]
dff
# Max date means the last day the device got checked.
# If the max day is ahead of failure date, it means this device returned to use after failed because got fixed.


# In[18]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize=(15,5))
fig.add_subplot(3, 2, 1) 
plt.plot(df.loc[df['device']=='S1F136J0',['failure','month']]['month'],df.loc[df['device']=='S1F136J0',         ['failure','month']]['failure'],         color = 'red')
fig.add_subplot(3, 2, 2) 
plt.plot(df.loc[df['device']=='W1F0KCP2',['failure','month']]['month'],df.loc[df['device']=='W1F0KCP2',         ['failure','month']]['failure'],         color = 'red')
fig.add_subplot(3, 2, 3)
plt.plot(df.loc[df['device']=='W1F0M35B',['failure','month']]['month'],df.loc[df['device']=='W1F0M35B',         ['failure','month']]['failure'],         color = 'red')
fig.add_subplot(3, 2, 4)
plt.plot(df.loc[df['device']=='S1F0GPFZ',['failure','month']]['month'],df.loc[df['device']=='S1F0GPFZ',         ['failure','month']]['failure'],         color = 'red')
fig.add_subplot(3, 2, 5)
plt.plot(df.loc[df['device']=='W1F11ZG9',['failure','month']]['month'],df.loc[df['device']=='W1F11ZG9',         ['failure','month']]['failure'],         color = 'red')


# In[19]:


df[df.device == 'S1F136J0']


# In[20]:


# Reducing the data set with unique device id
df.metric1.nunique()


# In[21]:


# Keeping the last record, as it comes with the most usefull infomation.
df1 = df.groupby('device').agg({'date':max})


# In[22]:


df1.shape


# In[23]:


df1=df1.reset_index()

df=df.reset_index(drop=True) 

df2= pd.merge(df1,df,how='left',on=['device','date'])

df2.shape


# In[24]:


df2.tail()


# In[25]:


#Creating feature called 'failure_before'
#If we just take the last record for the devices, we may lose information from those come back after failed ones

df2['failure_before']=0


# In[26]:


df2.loc[df2.device == 'S1F136J0','failure_before'] = 1
df2.loc[df2.device == 'W1F0KCP2','failure_before'] = 1
df2.loc[df2.device == 'W1F0M35B','failure_before'] = 1
df2.loc[df2.device == 'S1F0GPFZ','failure_before'] = 1
df2.loc[df2.device == 'W1F11ZG9','failure_before'] = 1


# In[27]:


#Redefining device Id value
df2.device


# In[28]:


Id = df2.device.values.tolist()


# In[29]:


# Changing device id values to the first four characters
Id1 = [] 
for i in Id:
    i = i[:4]
    Id1.append(i)

df2.device=Id1

df2.device.value_counts()


# In[30]:


get_ipython().run_line_magic('matplotlib', 'inline')
dev=pd.crosstab(df2['device'],df2['failure']) 

dev.div(dev.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
# Devices ID begins with ZIF0 fails the most, then W1F1 second


# In[31]:


#Data transformation
#Redefining data types
#redefining data type for some of the numerical features
cat_ftrs = ['metric3','metric4', 'metric5', 'metric7', 'metric9'] 
for col in cat_ftrs:
    df2[col]=df2[col].astype('object')


# In[32]:


# changing of activedays datatype to numerical
def str_to_num(str):
    return str.split(' ')[0]

df2.info()


# In[33]:


df2.activedays = df2.activedays.astype('str')

df2.activedays=df2.activedays.apply(str_to_num)
df2.activedays = df2.activedays.astype('int')
df2.info()


# In[34]:


for col in ['month','week_day']:
    df2[col]=df2[col].astype('object')


# In[35]:


# Data standarlization
# Numerically featuring normalization
f, axarr = plt.subplots(1,2) 
sns.distplot(df2['metric1'],ax=axarr[0]) 
axarr[0].set_title('Skewed Distribution') 
sns.distplot(np.log(1+df2['metric1']),ax=axarr[1]) 
axarr[1].set_title('Log-Transformed Distribution')


# In[36]:


f, axarr = plt.subplots(1,2) 

sns.distplot(df2['metric2'],ax=axarr[0]) 
axarr[0].set_title('Skewed Distribution') 
sns.distplot(np.log(1+df2['metric2']),ax=axarr[1]) 
axarr[1].set_title('Log-Transformed Distribution')


# In[37]:


f, axarr = plt.subplots(1,2) 
sns.distplot(df2['metric6'],ax=axarr[0]) 
axarr[0].set_title('Skewed Distribution') 
sns.distplot(np.log(1+df2['metric6']),ax=axarr[1]) 
axarr[1].set_title('Log-Transformed Distribution')
# It seems the data get more skewed after log, so not taking log on them.


# In[38]:


#numerical features standardization
num_ftrs =['metric1','metric2','metric6'] 
df2[num_ftrs]=scaler.fit_transform(df2[num_ftrs])

df2.info()


# In[39]:


# Dropping unimportant and redundant features
get_ipython().run_line_magic('matplotlib', 'inline')
sns.pairplot(df)
# metric7 and metric8 are highly linear related or equal to each other


# In[40]:


(df['metric7']==df['metric8']).value_counts()


# In[41]:


# Dropping attribute 8, as it is duplicated.
df.drop('metric8',axis=1,inplace=True)


# In[42]:


df.head()


# In[43]:


df2.drop(['date','max_date'],axis=1,inplace=True)
df2.info()


# In[44]:


# Getting dummies on categorical feature
df2.head()


# In[45]:


df2 = pd.get_dummies(df2,drop_first=True)
df2.shape


# In[46]:


df2.failure.value_counts()


# In[47]:


# Feature Selection
# Defining dependent and independent values
X = df2.drop('failure',axis=1)
Y = df2.failure


# In[48]:


get_ipython().run_line_magic('matplotlib', 'inline')
#Feature Selection
clf = RandomForestClassifier(n_estimators=50, max_features='auto')
clf= clf.fit(X,Y)

features = pd.DataFrame()
features['feature']= X.columns
features['important']=clf.feature_importances_
features.sort_values(by=['important'], ascending=False,inplace=True)
features.set_index('feature', inplace=True)
features.iloc[:20,:].plot(kind='barh', figsize=(30,30))


# In[49]:


model = SelectFromModel(clf,prefit=True)
x_reduced = model.transform(X)
print (x_reduced.shape)


# In[50]:


type(x_reduced)


# In[51]:


x_reduced=pd.DataFrame(x_reduced)


# In[52]:


x_reduced.head()


# In[53]:


x_reduced.info()


# In[54]:


#Resampling the data-test

# import model for imbalanced data set
from imblearn.over_sampling import RandomOverSampler
from imblearn.metrics import geometric_mean_score, make_index_balanced_accuracy, classification_report_imbalanced
from sklearn.metrics import confusion_matrix


# In[55]:


# testing on RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(x_reduced, Y, train_size=0.8,                                                     random_state=42)


rus = RandomOverSampler(random_state=42)
#From the training data craeated,upsampling would be done on the failed devices using the RandomOverSampler. This method object is to over-sample the minority class(es) by picking samples at random with replacement.

X_res, y_res = rus.fit_sample(X_train, y_train)
X_res.shape

X_res = pd.DataFrame(X_res)
#After upsampling, building of a random forest model to classify the failed devices would be done.

rf = RandomForestClassifier(n_estimators=5000, random_state=21)

a = rf.fit(X_res,y_res)



rf_test_pred = rf.predict(X_test)
rf_test_cm = confusion_matrix(y_test, rf_test_pred)
rf_test_cm

accuracy_score(y_test, rf_test_pred)

print(classification_report_imbalanced(y_test,rf_test_pred))


# In[56]:


# Model Training
# Oversampling before cross validate
log=LogisticRegression()
k=KNeighborsClassifier()
gbc =GradientBoostingClassifier()
rgr = RandomForestRegressor(n_estimators=100)
svc = SVC()
rfc = RandomForestClassifier(n_estimators=10)
xg_reg = XGBClassifier(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,                    hidden_layer_sizes=(5, 2), random_state=1)
algorithms=[k,log,gbc,rgr,svc,rfc,xg_reg,clf]
names=['KNeighborsClassifier','Logistic','GradientBoost','RandomForest','SVC','RandomForestCl','xgboost','neunet']


# In[57]:


iba = make_index_balanced_accuracy(alpha=0.1, squared=True)(geo)


# In[58]:


def cross_vali_fit_pred_1(X_res, y_res, algorithms = algorithms, names = names):
    # fit the data
    #x_train_reduced, x_test_reduced, y_train, y_test = train_test_split(x_reduced,Y,test_size=0.1, random_state=13)
    X_res = X_res.to_numpy()
    Geo_score = []
    Iba_score = []
    Accuracy = []
    F1 = []
    Recall = []
    Prec = []
    for i in range(len(algorithms)):
        j=1
        kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
        geo_score = []
        iba_score = []
        accuracy = []
        f1 = []
        recall = []
        prec = []
        for train_index,test_index in kf.split(X_res,y_res):
            xtr,xvd=X_res[train_index],X_res[test_index]
            ytr,yvd=y_res[train_index],y_res[test_index]
            algorithms[i] = algorithms[i].fit(xtr,ytr)
            y_pred_test = algorithms[i].predict(xvd).round()
            accuracy.append(accuracy_score(yvd, y_pred_test))
            geo_score.append(geo(yvd, y_pred_test))
            iba_score.append(iba(yvd, y_pred_test))
            f1.append(f1_score(yvd, y_pred_test,average='macro'))
            recall.append(recall_score(yvd, y_pred_test,average='macro'))
            prec.append(precision_score(yvd, y_pred_test))
            j +=1
        mean_ac = np.mean(accuracy)
        mean_geo = np.mean(geo_score)
        mean_f1 = np.mean(f1)
        mean_iba = np.mean(iba_score)
        mean_recall = np.mean(recall)
        mean_prec = np.mean(prec)
        F1.append(mean_f1)
        Geo_score.append(mean_geo)
        Iba_score.append(mean_iba)
        Accuracy.append(mean_ac)
        Recall.append(mean_recall)
        Prec.append(mean_prec)
        #cm=confusion_matrix(y_test,y_test_pred)
        #print(cm)
    metrics = pd.DataFrame(columns = ['Accuracy','geo_score','iba_score','f1','recall','prec'],index=names)
    metrics['Accuracy']=Accuracy
    metrics['geo_score']=Geo_score
    metrics['iba_score']=Iba_score
    metrics['f1']=F1
    metrics['recall']=Recall
    metrics['prec'] =Prec
    return metrics.sort_values('geo_score',ascending=False)


# In[59]:


cross_vali_fit_pred_1(X_res, y_res, algorithms = algorithms, names = names)


# In[60]:


RandomForestClassifier().get_params()


# In[61]:


geo(y_test, clf.predict(X_test).round())


# In[62]:


gbc.get_params()


# In[63]:


geo(y_test, gbc.predict(X_test).round())


# In[64]:


k.get_params()


# In[65]:


geo(y_test, k.predict(X_test).round())
# results being too good for true


# In[66]:


# Oversample within cross validation
# Oversampling done on only the training set. In this way the validation data will not have the same observations as in training set.
def cross_vali_fit_pred_2(X_train, y_train, algorithms = algorithms, names = names):
    # fit the data
    #x_train_reduced, x_test_reduced, y_train, y_test = train_test_split(x_reduced,Y,test_size=0.1, random_state=13)
    #X_test=X_test.to_numpy()
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()

    #y_test=y_test.to_numpy()
    Geo_score = []
    Iba_score = []
    Accuracy = []
    F1 = []
    Recall = []
    Prec = []
    for i in range(len(algorithms)):
        j=1
        kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
        geo_score = []
        iba_score = []
        accuracy = []
        f1 = []
        recall = []
        prec = []
        for train_index,test_index in kf.split(X_train, y_train):
            xtr,xvd=X_train[train_index],X_train[test_index]
            ytr,yvd=y_train[train_index],y_train[test_index]
            xtr_res,ytr_res=rus.fit_sample(xtr, ytr)
            algorithms[i] = algorithms[i].fit(xtr_res,ytr_res)
            y_pred_test = algorithms[i].predict(xvd).round()
            accuracy.append(accuracy_score(yvd, y_pred_test))
            geo_score.append(geo(yvd, y_pred_test))
            iba_score.append(iba(yvd, y_pred_test))
            f1.append(f1_score(yvd, y_pred_test,average='macro'))
            recall.append(recall_score(yvd, y_pred_test,average='macro'))
            prec.append(precision_score(yvd, y_pred_test))
            j +=1
            mean_ac = np.mean(accuracy)
        mean_geo = np.mean(geo_score)
        mean_f1 = np.mean(f1)
        mean_iba = np.mean(iba_score)
        mean_recall = np.mean(recall)
        mean_prec = np.mean(prec)
        F1.append(mean_f1)
        Geo_score.append(mean_geo)
        Iba_score.append(mean_iba)
        Accuracy.append(mean_ac)
        Recall.append(mean_recall)
        Prec.append(mean_prec)
        
    metrics = pd.DataFrame(columns = ['Accuracy','geo_score','iba_score','f1','recall','prec'],index=names)
    metrics['Accuracy']=Accuracy
    metrics['geo_score']=Geo_score
    metrics['iba_score']=Iba_score
    metrics['f1']=F1
    metrics['recall']=Recall
    metrics['prec']=Prec
    return metrics.sort_values('geo_score',ascending=False)


# In[67]:


type(X_train), type(y_train)


# In[68]:


cross_vali_fit_pred_2(X_train, y_train, algorithms = algorithms, names = names)


# In[69]:


#Model evaluation
X_test=X_test.to_numpy()

y_test=y_test.to_numpy()
probs1 = k.predict_proba(X_test)
probs2 = xg_reg.predict_proba(X_test)
probs3 = gbc.predict_proba(X_test)
probs4 = rfc.predict_proba(X_test)
probs1 = probs1[:,1]
probs2 = probs2[:,1]
probs3 = probs3[:,1]
probs4 = probs4[:,1]
#(separate from the ipykernel package so we can avoid doing imports until required)


# In[70]:


get_ipython().run_line_magic('matplotlib', 'inline')
fpr1, tpr1, thresholds1 = roc_curve(y_test, probs1)
fpr2, tpr2, thresholds2 = roc_curve(y_test, probs2)
fpr3, tpr3, thresholds3 = roc_curve(y_test, probs3)
fpr4, tpr4, thresholds4 = roc_curve(y_test, probs4)
roc_auc1 = auc(fpr1, tpr1)
roc_auc2 = auc(fpr2, tpr2)
roc_auc3 = auc(fpr3, tpr3)
roc_auc4 = auc(fpr4, tpr4)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr1, tpr1, marker='.', markerfacecolor='r',label='Kneighbor (area = %0.2f)' % roc_auc1)
plt.plot(fpr2, tpr2, marker='.', markerfacecolor='g',label='xgboost (area = %0.2f)' % roc_auc2)
plt.plot(fpr3, tpr3, marker='.', markerfacecolor='b',label='gradientboost (area = %0.2f)' % roc_auc3)
plt.plot(fpr4, tpr4, marker='.',markerfacecolor='y',label='randomforestclassifier (area = %0.2f)' % roc_auc4)
plt.xlabel('False Positive Rate(1-Specificity)')
plt.ylabel('True Positive Rate(Sensitivity)')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[71]:


geo(y_test, k.predict(X_test).round())


# In[72]:


geo(y_test, gbc.predict(X_test).round())


# In[73]:


geo(y_test, svc.predict(X_test).round())


# In[74]:


geo(y_test, clf.predict(X_test).round())#neutral network result is 0?


# In[75]:


geo(y_test, xg_reg.predict(X_test).round())


# In[76]:


geo(y_test, rfc.predict(X_test).round())


# In[77]:


#Model tunning
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()


# In[78]:


Geo1=[]
Geo2=[]
estimators = [i for i in range(5,150,10)]
Learning_rate = [i for i in np.arange(0.05,0.4,0.03)]
geo_score = []
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for i in estimators:
    for train_index,test_index in kf.split(X_train, y_train):
        xtr,xvd=X_train[train_index],X_train[test_index]
        ytr,yvd=y_train[train_index],y_train[test_index]
        xtr_res,ytr_res=rus.fit_sample(xtr, ytr)
        gbc =GradientBoostingClassifier(n_estimators=i).fit(xtr_res,ytr_res)
        y_pred_test = gbc.predict(xvd).round()
        #false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_test)
        geo_score.append(precision_score(yvd, y_pred_test))
    Geo1.append(np.mean(geo_score))

print(Geo1)


# In[79]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(estimators,Geo1)


# In[80]:


Geo2=[]
Learning_rate = [i for i in np.arange(0.01,0.2,0.03)]
geo_score = []
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for i in Learning_rate:
    for train_index,test_index in kf.split(X_train, y_train):
        xtr,xvd=X_train[train_index],X_train[test_index]
        ytr,yvd=y_train[train_index],y_train[test_index]
        xtr_res,ytr_res=rus.fit_sample(xtr, ytr)
        gbc =GradientBoostingClassifier(n_estimators=100,learning_rate=i).fit(xtr_res,ytr_res)
        y_pred_test = gbc.predict(xvd).round()
        #false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_test)
        geo_score.append(geo(yvd, y_pred_test))
    Geo2.append(np.mean(geo_score))

print(Geo2)


# In[81]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(Learning_rate,Geo2)


# In[82]:


#Deployment
gbc =GradientBoostingClassifier(n_estimators=100,learning_rate=0.1)


# In[83]:


Geo=[]
prec=[]
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for n_index,test_index in kf.split(X_train, y_train):
    xtr,xvd=X_train[train_index],X_train[test_index]
    ytr,yvd=y_train[train_index],y_train[test_index]
    xtr_res,ytr_res=rus.fit_sample(xtr, ytr)
    gbc =gbc.fit(xtr_res,ytr_res)
    y_pred_test = gbc.predict(xvd).round()
    
    #false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_test)
    Geo.append(geo(yvd, y_pred_test))
    prec.append(precision_score(yvd, y_pred_test))

print(np.mean(Geo))


# In[84]:


np.mean(prec)


# In[85]:


prob=gbc.predict_proba(X_test)


# In[86]:


prob


# In[87]:


y_test


# In[88]:


y_pred_prob = pd.DataFrame(y_test)
y_pred_prob['0']=prob[:,0]

y_pred_prob['1']=prob[:,1]

y_pred_prob.head(20)


# In[6]:



def main():
    date_rng = pd.date_range(start='2015-11-02', end='2015-01-01', freq='s')
    df = pd.DataFrame(date_rng, columns=['date'])

    np.random.seed(42)
    df['failure'] = np.random.randint(0, 100, size=(len(date_rng)))
    df = df.sample(frac=0.5, random_state=42).sort_values(by=['date'])
    df.to_csv('predictive_maintenance.csv', index=False)
    return


if __name__ == "__main__":
    main()


# In[7]:



while True:

        try:

            if  main is True:
                line1 = next(rdr, None)
                date, failure = line1[0], float(line1[1])
                # Converting csv columns to key value pair
                result = {}
                result[date] = failure
                # Converting dict to json as data format
                jresult = json.dumps(result)
                main = False

                producer.produce(topic, key=p_key, value=jresult, callback=acked)

            else:
                line = next(rdr, None)
                d1 = parse(date)
                d2 = parse(line[0])
                diff = ((d2 - d1).total_seconds())/args.speed
                time.sleep(diff)
                date, failure = line[0], float(line[1])
                result = {}
                result[date] = failure
                jresult = json.dumps(result)

                producer.produce(topic, key=p_key, value=jresult, callback=acked)

            producer.flush()

        except TypeError:
            sys.exit()


if __name__ == "__main__":
    main()


# In[ ]:




