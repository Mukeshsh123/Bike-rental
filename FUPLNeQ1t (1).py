#!/usr/bin/env python
# coding: utf-8

# # Bike Rental Prediction

# In[1]:


# Load the required libraries for analysis of data
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Set working directory
os.chdir("D:/mukesh")

# lets Check working directory
os.getcwd()


# In[3]:


# Load the data
Bike_Data = pd.read_csv("day.csv")


# # Explore the data

# In[4]:


# Check the dimensions(no of rows and no of columns)
Bike_Data.shape


# In[5]:


# Check names of dataset
Bike_Data.columns

# Rename variables in dataset
Bike_Data = Bike_Data.rename(columns = {'instant':'index','dteday':'date','yr':'year','mnth':'month','weathersit':'weather',
                                        'temp':'temperature','hum':'humidity','cnt':'count'})

Bike_Data.columns


# In[6]:


Bike_Data.columns


# In[7]:


#lets see first five observations of our data
Bike_Data.head()


# In[41]:


# lets see last five observations of our data
Bike_Data.tail()


# In[42]:


# Lets see the datatypes of the given data
Bike_Data.dtypes


# In[10]:


# lets Check summary of the dataset 
Bike_Data.describe()


# In[9]:


# Variable Identification 
Bike_Data['count'].dtypes


# In[8]:


#lets drop some variables because it doesnot carry any useful information

Bike_Data = Bike_Data.drop(['casual','registered','index','date'],axis=1)

# Lets check dimensions of data after removing some variables
Bike_Data.shape


# In[9]:


# Continous Variables 
cnames= ['temperature', 'atemp', 'humidity', 'windspeed', 'count']

# Categorical variables-
cat_cnames=['season', 'year', 'month', 'holiday', 'weekday', 'workingday','weather']


# # EDA or Data Preprocessing

# In[16]:


# Missing Value anlysis

# to check if there is any missing values
Missing_val = Bike_Data.isnull().sum()
Missing_val
# In our dataset we dont have any missing values.so that we dont need to do any imputation methods 


# In[8]:


# Outlier Analysis

# Lets save copy of dataset before preprocessing
df = Bike_Data.copy()
Bike_Data = df.copy() 

# Using seaborn library, we can viualize the outliers by plotting box plot
for i in cnames:
    print(i)
    sns.boxplot(y=Bike_Data[i])
    plt.xlabel(i)
    plt.ylabel("values")
    plt.title("Boxplot of "+i)
    plt.show()
    
# From boxplot we can see inliers in humidity and outliers in windspeed


# In[9]:


# Lets detect and remove outliers
for i in cnames:
    print(i)
    # Quartiles and IQR
    q25,q75 = np.percentile(Bike_Data[i],[25,75])
    IQR = q75-q25
    
    # Lower and upper limits 
    Minimum = q25 - (1.5 * IQR)
    print(Minimum)
    Maximum = q75 + (1.5 * IQR)
    print(Maximum)
    
    Minimum = Bike_Data.loc[Bike_Data[i] < Minimum ,i] 
    Maximum = Bike_Data.loc[Bike_Data[i] > Maximum ,i]

#we substituted minimum values for inliers and maximum values for outliers.
#from that we removed all the outliers.   


# In[10]:


# after replacing the outliers,let us plot boxplot for understanding
for i in cnames:
    print(i)
    sns.boxplot(y=Bike_Data[i])
    plt.xlabel(i)
    plt.ylabel("values")
    plt.title("Boxplot of "+i)
    plt.show()


# # Visualization

# In[11]:


# Univariate Analysis 

# temperature 
sns.FacetGrid(Bike_Data , height = 5).map(sns.distplot,'temperature').add_legend()
#normally distributed


# In[12]:


# humidity 
sns.FacetGrid(Bike_Data , height = 5).map(sns.distplot,'humidity').add_legend()
#normally distributed


# In[35]:


# windspeed
sns.FacetGrid(Bike_Data , height = 5).map(sns.distplot,'windspeed').add_legend()
#normally distributed


# In[15]:


#atemp
sns.FacetGrid(Bike_Data , height = 5).map(sns.distplot,'atemp').add_legend()
#normally distributed


# In[37]:


# count
sns.FacetGrid(Bike_Data , height = 5).map(sns.distplot,'count').add_legend()
#normally distributed


# In[11]:


# Lets check impact of continous variables on target variable

# count vs temperatur
plt.scatter(x='temperature',y='count',data=Bike_Data)
plt.title('Scatter plot count vs temperature')
plt.ylabel('count')
plt.xlabel('temperature')
plt.show()

#temperature is directly proportional to each other
#as temperature increases bike rental count also increases


# In[12]:


# count vs atemp
plt.scatter(x='atemp',y='count',data=Bike_Data)
plt.title('Scatter plot count vs atemp')
plt.ylabel('count')
plt.xlabel('atemp')
plt.show()

#as atemp increases bike rental count also increases


# In[13]:


# count vs humidity
plt.scatter(x='humidity',y='count',data=Bike_Data)
plt.title('Scatter plot count vs humidity')
plt.ylabel('count')
plt.xlabel('humidity')
plt.show()

# Apart from humidity,Bike rental count does not get affected


# In[14]:


# count vs windspeed
plt.scatter(x='windspeed',y='count',data=Bike_Data)
plt.title('Scatter plot count vs windspeed')
plt.ylabel('count')
plt.xlabel('windspeed')
plt.show()

# Apart from windspeed,Bike rental count does not get affected


# In[17]:


#for categorical variables


# SEASON
print(Bike_Data.groupby(['season'])['count'].sum())
#based on the season, bike rental count is high in season 3 which is fall and low in season 1 which is spring

#lets visualize the count using scatterplot
sns.scatterplot(x='season',y='count',data = Bike_Data)


# In[18]:


# YEAR
print(Bike_Data.groupby(['year'])['count'].sum())
#based on the year, bike rental count is high in the year 1 which is 2012

#lets visualize the count using scatterplot
sns.scatterplot(x='year',y='count',data = Bike_Data)


# In[19]:


# MONTH
print(Bike_Data.groupby(['month'])['count'].sum())
#Based on the month, Bike rental count is high in 8 which is in august and low in 1 which is in january

#lets visualize the count using scatterplot
sns.scatterplot(x='month',y='count',data = Bike_Data)


# In[20]:


#HOLIDAY
print(Bike_Data.groupby(['holiday'])['count'].sum())
#Based on the holiday, bike rental count is high in 0 which is holiday and low in 1 which is working day

#lets visualize the count using scatterplot
sns.scatterplot(x='holiday',y='count',data = Bike_Data)


# In[21]:


# WEAKDAY
print(Bike_Data.groupby(['weekday'])['count'].sum())
#Based on the weakday, bike rental count is high in 5 which is friday and low in 0 which is sunday

#lets visualize the count using scatterplot
sns.scatterplot(x='weekday',y='count',data = Bike_Data)


# In[22]:


# WORKINGDAY
print(Bike_Data.groupby(['workingday'])['count'].sum())
#Based on the workingday, Bike rental count is high in 1 which is working day and low in 0 which is hoiday

#lets visualize the count using scatterplot
sns.scatterplot(x='workingday',y='count',data = Bike_Data)


# In[66]:


#WEATHER
print(Bike_Data.groupby(['weather'])['count'].sum())
#Based n the weather bike rental count is higher in 1 which clear,few clouds,partly cloudy and there is no bikes rental in 4

#lets visualize the count using scatterplot
sns.scatterplot(x='weather',y='count',data = Bike_Data)


# In[16]:


# Bike rented with respected to tempeature and humidity
f, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(x="temperature", y="count",
                hue="humidity", size="count",
                palette="rainbow",sizes=(1, 100), linewidth=0,
                data=Bike_Data,ax=ax)
plt.title("Varation in bike rented with respect to temperature and humidity")
plt.ylabel("Bike rental count")
plt.xlabel("temperature")

# based on the below plot we know that bike rental is higher when the 
                            #temperature is between 0.4 to 0.8 
                            #humidity less than 0.8


# In[17]:


#Bikes rented with respect to temperature and windspeed
f, ax = plt.subplots(figsize=(10,10))
sns.scatterplot(x="temperature", y="count",
                hue="windspeed", size="humidity",
                palette="rainbow",sizes=(1, 100), linewidth=0,
                data=Bike_Data,ax=ax)
plt.title("Varation in bike rented with respect to  temperature and windspeed")
plt.ylabel("Bike rental count")
plt.xlabel("temperature")

#based on the below plot we know that bike rental is higher when the 
                            #temperature is between 0.4 to 0.8 
                            #humidity is less than 0.8
                            #windspeed is less than 0.2


# In[18]:


# Bikes rented with respect to temperature and season
f, ax = plt.subplots(figsize=(10,10))
sns.scatterplot(x="temperature", y="count",
                hue="season", size="count",style= "weather",
                palette="rainbow",sizes=(1, 100), linewidth=0,
                data=Bike_Data,ax=ax)
plt.title("Varation in bike rented with respect to temperature and season")
plt.ylabel("Bike rental count")
plt.xlabel("Normalized temperature")

#based on the below plot we know that bike rental is higher when the 
                            #temperature is between 0.4 to 0.8 
                            #season was 2 and 3
                            #weather was from 1 and 2


# # Feature Selection

# In[13]:


# Lets save dataset after outlier analysis 
df =  Bike_Data.copy()
Bike_Data = df.copy()


# In[22]:


# Correlation analysis

# Correlation matrix continuous variables
Bike_corr= Bike_Data.loc[:,cnames]

# Generate correlation matrix
corr_matrix = Bike_corr.corr()
(print(corr_matrix))


# In[23]:


# Set the width and hieght of the plot
f, ax = plt.subplots(figsize=(15,15))

#Plot using seaborn library
sns.heatmap(corr_matrix, mask=np.zeros_like(corr_matrix, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax,annot=True)

plt.title("Correlation Plot For Numeric or Continous Variables")

#from the below plot,we came to know that both temperature and atemp variables are carrying almost same information
#hence there is no need to continue with both variables.
#so we need to drop any one of the variables
#here I am dropping atemp variables


# In[26]:


import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[25]:


# ANOVA test for categorical variables

for i in cat_cnames:
    mod = ols('count' + '~' + i, data = Bike_Data).fit()
    aov_table = sm.stats.anova_lm(mod, typ = 2)
    print(aov_table)


# In[ ]:


#based on the anova result, we are going to drop three variables namely,
                            #HOLIDAY
                            #WEEKDAY
                            #WORKINGDAY
            #because these variables having the p-value > 0.05


# In[23]:


# Removing the variables which has p-value > 0.05 and correlated variable
Bike_Data = Bike_Data.drop(['atemp', 'holiday','weekday','workingday'], axis=1)


# In[24]:


# After removing variables lets check dimension of the data
Bike_Data.shape


# In[16]:


# After removing variables lets check column names of the data
Bike_Data.columns


# In[25]:


#after removing the variables, we need update numerical and categorical variables

# numerical variable
cnames = ['temperature','humidity', 'windspeed', 'count']

# Categorical variables
catnames = ['season', 'year', 'month','weather']


# # Feature scaling

# In[ ]:


#based on the details of the attributes given, all the numerical variables are normalised


# In[27]:


#lets visualise the numerical variables to see normality
for i in cnames:
    print(i)
    sm.qqplot(Bike_Data[i])
    plt.title("Normalized qq plot for " +i)
    plt.show()


# In[32]:


Bike_Data.describe()


# In[ ]:


#we confirmed the normalized data based on the qqplot and summary of the data


# # Model Development

# In[18]:


# Load Required libraries for model development 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# In[19]:


#In Regression problems, we can't pass directly categorical variables.
#so we need to convert all categorical variables into dummy variables

df = Bike_Data
Bike_Data = df

#  Converting categorical variables to dummy variables
Bike_Data = pd.get_dummies(Bike_Data,columns=catnames)


# In[20]:


Bike_Data.shape


# In[21]:


Bike_Data.columns


# In[22]:


# Lets Divide the data into train and test set 

X= Bike_Data.drop(['count'],axis=1)
y= Bike_Data['count']


# In[23]:


# Divide data into train and test sets
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=.20)


# In[24]:


# Function for Error metrics to calculate the performance of model
def MAPE(y_true,y_prediction):
    mape= np.mean(np.abs(y_true-y_prediction)/y_true)*100
    return mape


# In[26]:


# Linear Regression model


# In[27]:


# Import libraries
import statsmodels.api as sm

# Linear Regression model
LinearRegression_model= sm.OLS(y_train,X_train).fit()
print(LinearRegression_model.summary())


# In[28]:


# Model prediction on  train data
LinearRegression_train= LinearRegression_model.predict(X_train)

# Model prediction on test data
LinearRegression_test= LinearRegression_model.predict(X_test)

# Model performance on train data
MAPE_train= MAPE(y_train,LinearRegression_train)

# Model performance on test data
MAPE_test= MAPE(y_test,LinearRegression_test)

# r2 value for train data
r2_train= r2_score(y_train,LinearRegression_train)

# r2 value for test data-
r2_test=r2_score(y_test,LinearRegression_test)

# RMSE value for train data
RMSE_train = np.sqrt(metrics.mean_squared_error(y_train,LinearRegression_train))

# RMSE value for test data
RMSE_test = np.sqrt(metrics.mean_squared_error(y_test,LinearRegression_test))

print("Mean Absolute Precentage Error for train data="+str(MAPE_train))
print("Mean Absolute Precentage Error for test data="+str(MAPE_test))
print("R^2_score for train data="+str(r2_train))
print("R^2_score for test data="+str(r2_test))
print("RMSE for train data="+str (RMSE_train))
print("RMSE for test data="+str(RMSE_test))


# In[29]:


Error_MetricsLT = {'Model Name': ['Linear Regression'],
                 'MAPE_Train':[MAPE_train],
                 'MAPE_Test':[MAPE_test],
                 'R-squared_Train':[r2_train],
                 'R-squared_Test':[r2_test],
                 'RMSE_train':[RMSE_train],
                 'RMSE_test':[RMSE_test]}

LinearRegression_Results = pd.DataFrame(Error_MetricsLT)


# In[30]:


LinearRegression_Results


# In[31]:


#Decision tree model


# In[32]:


# Lets Build decision tree model on train and test data
from sklearn.tree import DecisionTreeRegressor

# Decision tree for regression
DecisionTree_model= DecisionTreeRegressor(max_depth=2).fit(X_train,y_train)

# Model prediction on train data
DecisionTree_train= DecisionTree_model.predict(X_train)

# Model prediction on test data
DecisionTree_test= DecisionTree_model.predict(X_test)

# Model performance on train data
MAPE_train= MAPE(y_train,DecisionTree_train)

# Model performance on test data
MAPE_test= MAPE(y_test,DecisionTree_test)

# r2 value for train data
r2_train= r2_score(y_train,DecisionTree_train)

# r2 value for test data
r2_test=r2_score(y_test,DecisionTree_test)

# RMSE value for train data
RMSE_train = np.sqrt(metrics.mean_squared_error(y_train,DecisionTree_train))

# RMSE value for test data
RMSE_test = np.sqrt(metrics.mean_squared_error(y_test,DecisionTree_test))

print("Mean Absolute Precentage Error for train data="+str(MAPE_train))
print("Mean Absolute Precentage Error for test data="+str(MAPE_test))
print("R^2_score for train data="+str(r2_train))
print("R^2_score for test data="+str(r2_test))
print("RMSE for train data="+str(RMSE_train))
print("RMSE for test data="+str(RMSE_test))


# In[33]:


Error_MetricsDT = {'Model Name': ['Decision Tree'],
                 'MAPE_Train':[MAPE_train],
                 'MAPE_Test':[MAPE_test],
                 'R-squared_Train':[r2_train],
                 'R-squared_Test':[r2_test],
                 'RMSE_train':[RMSE_train],
                 'RMSE_test':[RMSE_test]}
                   
DecisionTree_Results = pd.DataFrame(Error_MetricsDT)


# In[34]:


DecisionTree_Results


# In[35]:


# Random Search CV In Decision Tree


# In[36]:


# Import libraries 
from sklearn.model_selection import RandomizedSearchCV

RandomDecisionTree = DecisionTreeRegressor(random_state = 0)
depth = list(range(1,20,2))
random_search = {'max_depth': depth}

# Lets build a model using above parameters on train data 
RandomDecisionTree_model= RandomizedSearchCV(RandomDecisionTree,param_distributions= random_search,n_iter=5,cv=5)
RandomDecisionTree_model= RandomDecisionTree_model.fit(X_train,y_train)


# In[37]:


# Lets look into best fit parameters
best_parameters = RandomDecisionTree_model.best_params_
print(best_parameters)


# In[38]:


# Again rebuild decision tree model using randomsearch best fit parameter ie
# with maximum depth = 7
RDT_best_model = RandomDecisionTree_model.best_estimator_
print(RDT_best_model)


# In[39]:


# Prediction on train data 
RDT_train = RDT_best_model.predict(X_train)

# Prediction on test data 
RDT_test = RDT_best_model.predict(X_test)

# Lets check Model performance on both test and train using error metrics of regression like mape,rsquare value
# MAPE for train data 
MAPE_train= MAPE(y_train,RDT_train)

# MAPE for test data 
MAPE_test= MAPE(y_test,RDT_test)

# Rsquare for train data
r2_train= r2_score(y_train,RDT_train)

# Rsquare for test data
r2_test=r2_score(y_test,RDT_test)

# RMSE value for train data
RMSE_train = np.sqrt(metrics.mean_squared_error(y_train,RDT_train))

# RMSE value for test data
RMSE_test = np.sqrt(metrics.mean_squared_error(y_test,RDT_test))


# Lets print the results 
print("Best Parameter="+str(best_parameters))
print("Best Model="+str(RDT_best_model))
print("Mean Absolute Precentage Error for train data="+str(MAPE_train))
print("Mean Absolute Precentage Error for test data="+str(MAPE_test))
print("R^2_score for train data="+str(r2_train))
print("R^2_score for test data="+str(r2_test))
print("RMSE for train data="+str (RMSE_train))
print("RMSE for test data="+str(RMSE_test))


# In[40]:


Error_MetricsRDT = {'Model Name': ['Random Search CV Decision Tree'],
                 'MAPE_Train':[MAPE_train],
                 'MAPE_Test':[MAPE_test],
                 'R-squared_Train':[r2_train],
                 'R-squared_Test':[r2_test],
                 'RMSE_train':[RMSE_train],
                 'RMSE_test':[RMSE_test]}
                   
RandomDecisionTree_Results = pd.DataFrame(Error_MetricsRDT)


# In[41]:


RandomDecisionTree_Results


# In[42]:


# Grid Search CV in Decision Tree


# In[43]:


# Import libraries
from sklearn.model_selection import GridSearchCV

GridDecisionTree= DecisionTreeRegressor(random_state=0)
depth= list(range(1,20,2))
grid_search= {'max_depth':depth}

# Lets build a model using above parameters on train data
GridDecisionTree_model= GridSearchCV(GridDecisionTree,param_grid=grid_search,cv=5)
GridDecisionTree_model= GridDecisionTree_model.fit(X_train,y_train)


# In[44]:


# Lets look into best fit parameters from gridsearch cv DT
best_parameters = GridDecisionTree_model.best_params_
print(best_parameters)


# In[45]:


# Again rebuild decision tree model using gridsearch best fit parameter ie
# with maximum depth = 7
GDT_best_model = GridDecisionTree_model.best_estimator_


# In[46]:


# Prediction on train data 
GDT_train = GDT_best_model.predict(X_train)

# Prediction on train data  test data-
GDT_test = GDT_best_model.predict(X_test)

# Lets check Model performance on both test and train using error metrics of regression like mape,rsquare value
# MAPE for train data 
MAPE_train= MAPE(y_train,GDT_train)

# MAPE for test data 
MAPE_test= MAPE(y_test,GDT_test)

# Rsquare for train data
r2_train= r2_score(y_train,GDT_train)

# Rsquare for train data
r2_test=r2_score(y_test,GDT_test)

# RMSE value for train data
RMSE_train = np.sqrt(metrics.mean_squared_error(y_train,GDT_train))

# RMSE value for test data
RMSE_test = np.sqrt(metrics.mean_squared_error(y_test,GDT_test))


print("Best Parameter="+str(best_parameters))
print("Best Model="+str(GDT_best_model))
print("Mean Absolute Precentage Error for train data="+str(MAPE_train))
print("Mean Absolute Precentage Error for test data="+str(MAPE_test))
print("R^2_score for train data="+str(r2_train))
print("R^2_score for test data="+str(r2_test))
print("RMSE for train data="+str (RMSE_train))
print("RMSE for test data="+str(RMSE_test))


# In[47]:


Error_MetricsGDT = {'Model Name': ['Grid Search CV Decision Tree'],
                 'MAPE_Train':[MAPE_train],
                 'MAPE_Test':[MAPE_test],
                 'R-squared_Train':[r2_train],
                 'R-squared_Test':[r2_test],
                 'RMSE_train':[RMSE_train],
                 'RMSE_test':[RMSE_test]}
                   
GridDecisionTree_Results = pd.DataFrame(Error_MetricsGDT)


# In[48]:


GridDecisionTree_Results


# In[49]:


# Random Forest


# In[50]:


# Import libraris
from sklearn.ensemble import RandomForestRegressor

# Random Forest for regression
RF_model= RandomForestRegressor(n_estimators=100).fit(X_train,y_train)

# Prediction on train data
RF_train= RF_model.predict(X_train)

# Prediction on test data
RF_test= RF_model.predict(X_test)

# MAPE For train data
MAPE_train= MAPE(y_train,RF_train)

# MAPE For test data
MAPE_test= MAPE(y_test,RF_test)

# Rsquare  For train data
r2_train= r2_score(y_train,RF_train)

# Rsquare  For test data
r2_test=r2_score(y_test,RF_test)

# RMSE value for train data
RMSE_train = np.sqrt(metrics.mean_squared_error(y_train,RF_train))

# RMSE value for test data
RMSE_test = np.sqrt(metrics.mean_squared_error(y_test,RF_test))

print("Mean Absolute Precentage Error for train data="+str(MAPE_train))
print("Mean Absolute Precentage Error for test data="+str(MAPE_test))
print("R^2_score for train data="+str(r2_train))
print("R^2_score for test data="+str(r2_test))
print("RMSE for train data="+str (RMSE_train))
print("RMSE for test data="+str(RMSE_test))


# In[51]:


Error_MetricsRF = {'Model Name': ['Random Forest'],
                 'MAPE_Train':[MAPE_train],
                 'MAPE_Test':[MAPE_test],
                 'R-squared_Train':[r2_train],
                 'R-squared_Test':[r2_test],
                 'RMSE_train':[RMSE_train],
                 'RMSE_test':[RMSE_test]}
                   
RandomForest_Results = pd.DataFrame(Error_MetricsRF)


# In[52]:


RandomForest_Results


# In[53]:


# Random Search CV in Random Forest


# In[54]:


# Import libraries
from sklearn.model_selection import RandomizedSearchCV

RandomRandomForest = RandomForestRegressor(random_state = 0)
n_estimator = list(range(1,100,2))
depth = list(range(1,20,2))
random_search = {'n_estimators':n_estimator, 'max_depth': depth}

# Lets build a model using above parameters on train data
RandomRandomForest_model= RandomizedSearchCV(RandomRandomForest,param_distributions= random_search,n_iter=5,cv=5)
RandomRandomForest_model= RandomRandomForest_model.fit(X_train,y_train)


# In[55]:


# Best parameters for model
best_parameters = RandomRandomForest_model.best_params_
print(best_parameters)


# In[56]:


# Again rebuild random forest  model using gridsearch best fit parameter
RRF_best_model = RandomRandomForest_model.best_estimator_


# In[57]:


# Prediction on train data
RRF_train = RRF_best_model.predict(X_train)

# Prediction on test data
RRF_test = RRF_best_model.predict(X_test)

# Lets check Model performance on both test and train using error metrics of regression like mape,rsquare value
# MAPE for train data 
MAPE_train= MAPE(y_train,RRF_train)

# MAPE for test data
MAPE_test= MAPE(y_test,RRF_test)

# Rsquare for train data
r2_train= r2_score(y_train,RRF_train)

# Rsquare for test data
r2_test=r2_score(y_test,RRF_test)

# RMSE value for train data
RMSE_train = np.sqrt(metrics.mean_squared_error(y_train,RRF_train))

# RMSE value for test data
RMSE_test = np.sqrt(metrics.mean_squared_error(y_test,RRF_test))


print("Best Parameter="+str(best_parameters))
print("Best Model="+str(RRF_best_model))
print("Mean Absolute Precentage Error for train data="+str(MAPE_train))
print("Mean Absolute Precentage Error for test data="+str(MAPE_test))
print("R^2_score for train data="+str(r2_train))
print("R^2_score for test data="+str(r2_test))
print("RMSE for train data="+str (RMSE_train))
print("RMSE for test data="+str(RMSE_test))


# In[58]:


Error_MetricsRSRF = {'Model Name': ['Random Search CV Random Forest'],
                 'MAPE_Train':[MAPE_train],
                 'MAPE_Test':[MAPE_test],
                 'R-squared_Train':[r2_train],
                 'R-squared_Test':[r2_test],
                 'RMSE_train':[RMSE_train],
                 'RMSE_test':[RMSE_test]}
                   
RandomSearchRandomForest_Results = pd.DataFrame(Error_MetricsRSRF)


# In[59]:


RandomSearchRandomForest_Results


# In[60]:


# Grid search CV in Random Forest


# In[61]:


# Import libraries
from sklearn.model_selection import GridSearchCV

GridRandomForest= RandomForestRegressor(random_state=0)
n_estimator = list(range(1,20,2))
depth= list(range(1,20,2))
grid_search= {'n_estimators':n_estimator, 'max_depth': depth}


# In[62]:


# Lets build a model using above parameters on train data using random forest grid search cv 
GridRandomForest_model= GridSearchCV(GridRandomForest,param_grid=grid_search,cv=5)
GridRandomForest_model= GridRandomForest_model.fit(X_train,y_train)


# In[63]:


# Best fit parameters for model
best_parameters_GRF = GridRandomForest_model.best_params_
print(best_parameters_GRF)


# In[64]:


# Again rebuild random forest model using gridsearch best fit parameter 
GRF_best_model = GridRandomForest_model.best_estimator_


# In[65]:


# Prediction on train data
GRF_train = GRF_best_model.predict(X_train)

# Prediction on test data
GRF_test = GRF_best_model.predict(X_test)

# Lets check Model performance on both test and train using error metrics of regression like mape,rsquare value
# MAPE for train data
MAPE_train= MAPE(y_train,GRF_train)

# MAPE for test data
MAPE_test= MAPE(y_test,GRF_test)

# Rsquare for train data
r2_train= r2_score(y_train,GRF_train)

# Rsquare for test data
r2_test=r2_score(y_test,GRF_test)

# RMSE value for train data
RMSE_train = np.sqrt(metrics.mean_squared_error(y_train,GRF_train))

# RMSE value for test data
RMSE_test = np.sqrt(metrics.mean_squared_error(y_test,GRF_test))

print("Best Parameter="+str(best_parameters))
print("Best Model="+str(GRF_best_model))
print("Mean Absolute Precentage Error for train data="+str(MAPE_train))
print("Mean Absolute Precentage Error for test data="+str(MAPE_test))
print("R^2_score for train data="+str(r2_train))
print("R^2_score for test data="+str(r2_test))
print("RMSE for train data="+str (RMSE_train))
print("RMSE for test data="+str(RMSE_test))


# In[66]:


Error_MetricsGSRF = {'Model Name': ['Grid search CV Random Forest'],
                 'MAPE_Train':[MAPE_train],
                 'MAPE_Test':[MAPE_test],
                 'R-squared_Train':[r2_train],
                 'R-squared_Test':[r2_test],
                 'RMSE_train':[RMSE_train],
                 'RMSE_test':[RMSE_test]}
                   
GridSearchRandomForest_Results = pd.DataFrame(Error_MetricsGSRF)


# In[67]:


GridSearchRandomForest_Results


# In[68]:


Final_Results = pd.concat([LinearRegression_Results,
                                DecisionTree_Results,
                                RandomDecisionTree_Results,
                                GridDecisionTree_Results,
                                RandomForest_Results,
                                RandomSearchRandomForest_Results,
                                GridSearchRandomForest_Results,], ignore_index=True, sort =False)


# In[69]:


Final_Results


# In[79]:


# From above results Random Forest model have optimum values and this
# algorithm is good for our data 

# Lets save the out put of finalized model (RF)

input = y_test.reset_index()
pred = pd.DataFrame(RF_test,columns = ['pred'])
Final_output = pred.join(input)


# In[80]:


Final_output


# In[81]:


Final_output.to_csv("Final_results_py.csv")


# In[ ]:




