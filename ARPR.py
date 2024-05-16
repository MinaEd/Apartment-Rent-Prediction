#!/usr/bin/env python
# coding: utf-8

# # Appartment Rent Prediction

# Libraries Importation

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import math
import re
import sklearn
from joblib import dump
from joblib import load
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score


# 1.Reading and Exploring the dataset

# In[2]:


df=pd.read_csv("ApartmentRentPrediction.csv")
df


# 2.Diving into the dataset (Analyzing each column)

# In[3]:


for column in df.columns:
    unique_values = df[column].nunique()
    print(f"{column}: {unique_values} unique values")


# In[4]:


####Checking the type of each column
df.dtypes


# In[5]:


###performing statistical analysis
df.describe()


# In[6]:


X = df.drop(['price_display'], axis=1)
Y = pd.DataFrame(df['price_display'])


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=42)


# 3.Preprocessing with different techniques

# I. Data Cleaning (Handling Missing Values, Duplicates, Incorrect formats (using different techniques in replacing or dropping))

# In[8]:


####Nulls
X_train.isna().sum()


# In[9]:


####Duplicates
duplicates = X_train[X_train.duplicated()]

if duplicates.empty:
    print("No duplicate rows found.")
else:
    print("Duplicate rows found:")
    print(duplicates)


# In[10]:


#bathrooms and bedrooms since there are low number of missing values mean or mode replacement or we can drop them
X_train['bedrooms']=X_train['bedrooms'].fillna(X_train['bedrooms'].mode().iloc[0])
X_train['bathrooms']=X_train['bathrooms'].fillna(X_train['bathrooms'].mode().iloc[0])


# In[11]:


X_train['pets_allowed']


# In[12]:


X_train['pets_allowed'] = X_train['pets_allowed'].fillna('No').apply(lambda x: 'Yes' if x != 'No' else x)
X_train['pets_allowed']
#Converting any pets to Yes and Missing Values to No
#Encode it to 0 and 1


# In[13]:


#Bathrooms has some floating points which must be integer so we round it up
X_train['bathrooms'] = X_train['bathrooms'].apply(lambda x: math.ceil(x))
X_train['bathrooms']


# In[14]:


#Bedrooms has no float values but its datatype float we convert it also
X_train['bedrooms'] = X_train['bedrooms'].astype(np.int64)


# In[15]:


####Long and Lat columns we can fill NaN with the mean.
X_train['longitude'] = X_train['longitude'].fillna(X_train['longitude'].mean())
X_train['latitude'] = X_train['latitude'].fillna(X_train['latitude'].mean())


# In[16]:


X_train['cityname']=X_train['cityname'].fillna(X_train['cityname'].mode().iloc[0])
X_train['state']=X_train['state'].fillna(X_train['state'].mode().iloc[0])


# In[17]:


###address
X_train['address']=X_train['address'].fillna("Unknown")


# In[18]:


X_train['amenities'] = X_train['amenities'].fillna(0)


# In[19]:


X_train.isna().sum()


# In[20]:


y_train['price_display']
#Format is not clear


# In[21]:


y_train['price_display'] = y_train['price_display'].str.replace(r'[^0-9.]', '', regex=True).astype(np.int64)
y_train['price_display']


# # Note : We must preprocess all columns even if we will drop them

# Processing numerical features 

# In[22]:


##id column can be mapped to index 0 till 8999 (useless it is integer anyways and unique and we won't use it)
##we just do it to make it look easier
unique_ids = X_train['id'].unique()
id_mapping = {original_id: new_id for new_id, original_id in enumerate(unique_ids)}
X_train['id'] = X_train['id'].map(id_mapping)
X_train['id']


# In[23]:


#### time column
max_time = X_train['time'].max()
min_time = X_train['time'].min()
print(f"Maximum time in : {max_time}")
print(f"Minimum time in : {min_time}")


# In[24]:


X_train['time']=X_train['time']/X_train['time'].min()


# In[25]:


X_train['time']


# In[26]:


X_train['bedrooms'].value_counts()[0]


# In[27]:


X_train['bedrooms']=X_train['bedrooms'].replace(0,1)


# Processing categorical features

# In[28]:


######Extracting Price Info from body column######
def extract_price(text):
    prices = re.findall(r'\$(\d+(?:\.\d+)?)', text)
    if prices:
        return prices[0]
    else:
        return 0

X_train['extracted_price'] = X_train['body'].apply(extract_price)


# In[29]:


def extract_price_df(text):
    if isinstance(text, str):  # Check if text is a string
        prices = re.findall(r'\$(\d+(?:\.\d+)?)', text)
        if prices:
            return prices[0]
    return 0


# In[30]:


X_train['extracted_price']=X_train['extracted_price'].astype(np.float64)
X_train['extracted_price']


# In[31]:


d1=X_train['cityname'].mode().iloc[0]
d2=X_train['state'].mode().iloc[0]
d3=X_train['state']
d4=X_train['cityname']
d5 = X_train['bathrooms'].mode().iloc[0]
dfff=X_train[X_train['cityname']==d1]
mostcity = dfff['extracted_price'].mean()
dffff=X_train[X_train['state']==d2]
moststate = dffff['extracted_price'].mean()
dfBath=X_train[X_train['bathrooms']==d5]
mostBath = dfBath['extracted_price'].mean()
d5=X_train['square_feet'].mean()
mostBath


# In[32]:


####amenities column we fill NaN with 0 and we count each amenity.
##  Ex: Gym , Parking , Pool = 3
def encode_amenities(value):
    items = value.split(',') if isinstance(value, str) else []
    count=len(items)
    return count


if X_train['amenities'].dtype == object:
    X_train['amenities'] = X_train['amenities'].apply(encode_amenities)


X_train['amenities']


# In[33]:


####Replacing nan with 0 then with the mean or mode which is equal 3.
X_train.loc[X_train['amenities'] == 0, 'amenities'] = 3
X_train['amenities']


# In[34]:


X_train['amenities'].value_counts()


# In[35]:


X_train['state'].value_counts()


# In[36]:


desired_state='LA'
state_df = X_train[X_train['state'] == desired_state]
max_price = state_df['extracted_price'].max()
min_price = state_df['extracted_price'].min()
print(f"Maximum price in {desired_state}: {max_price}")
print(f"Minimum price in {desired_state}: {min_price}")


# In[37]:


#grouping the state according to its names and replacing with the avg price of this state name
state_encoding = X_train.groupby('state')['extracted_price'].mean().reset_index()
state_encoding.columns = ['state', 'state_price_mean']
X_train = pd.merge(X_train, state_encoding, on='state', how='left')
X_train['state']=X_train['state_price_mean']
X_train.drop(['state_price_mean'], axis=1, inplace=True)


# In[38]:


#grouping the city according to its names and replacing with the avg price of this city name
city_encoding = X_train.groupby('cityname')['extracted_price'].mean().reset_index()
city_encoding.columns = ['cityname', 'city_price_mean']
X_train = pd.merge(X_train, city_encoding, on='cityname', how='left')
X_train['cityname']=X_train['city_price_mean']
X_train.drop(['city_price_mean'], axis=1, inplace=True)


# In[39]:


#####Renaming them######
X_train = X_train.rename(columns={'cityname': 'CityavgRentprice', 'state': 'StateavgRentprice'})


# In[40]:


X_train['source'].value_counts()


# In[41]:


X_train['price_type'].value_counts()
##### We convert them to Monthly


# In[42]:


X_train['category'].value_counts()
##### We convert them to housing/rent/apartment


# In[43]:


X_train['price_type'] = X_train['price_type'].replace({'Weekly': 'Monthly', 'Monthly|Weekly': 'Monthly'})
X_train['source'] = X_train['source'].replace({'rentbits': 'RentLingo', 'Real Estate Agent': 'RentLingo', 'tenantcloud': 'RentLingo', 'RENTCafé': 'RentLingo'})
X_train['category'] = X_train['category'].replace({'housing/rent/short_term': 'housing/rent/apartment', 'housing/rent/home': 'housing/rent/apartment'})
###After converting we label encode it and we will drop this column as well as it has 1 value


# II. Encoding (using the built-in LabelEncoder)

# In[44]:


label_encoder = LabelEncoder()
X_train['source'] = label_encoder.fit_transform(X_train['source'])
X_train['price_type'] = label_encoder.fit_transform(X_train['price_type'])
X_train['category'] = label_encoder.fit_transform(X_train['category'])
X_train['address'] = label_encoder.fit_transform(X_train['address'])
X_train['body'] = label_encoder.fit_transform(X_train['body'])
X_train['title'] = label_encoder.fit_transform(X_train['title'])
X_train['pets_allowed'] = label_encoder.fit_transform(X_train['pets_allowed'])
X_train['has_photo'] = label_encoder.fit_transform(X_train['has_photo'])
X_train['fee'] = label_encoder.fit_transform(X_train['fee'])
X_train['currency'] = label_encoder.fit_transform(X_train['currency'])
label_encoder_path = 'le.joblib'
dump(label_encoder, label_encoder_path)


# In[45]:


df.dtypes


# In[46]:


###df after encoding
X_train


# III. Assuring no outliers

# In[47]:


# Select the numerical columns for outlier detection
numerical_columns = ['bathrooms', 'bedrooms', 'square_feet', 'latitude', 'longitude']

fig, axes = plt.subplots(1, len(numerical_columns), figsize=(20, 5), sharey=True)

for idx, column in enumerate(numerical_columns):
    sns.boxplot(data=X_train, x=column, ax=axes[idx])
    
    axes[idx].set_title(f'Box Plot of {column}')


plt.tight_layout()


plt.show()


# In[48]:


# Select the numerical columns for outlier detection
numerical_columns = ['bathrooms', 'bedrooms', 'square_feet', 'latitude', 'longitude']
List1lower = []
List1upper =[]
for column in numerical_columns:
   
    z_scores = zscore(X_train[column])

    # Define a threshold for outliers (e.g., Z-score > 3 or < -3)
    threshold = 3

    # Identify outliers
    outliers = np.abs(z_scores) > threshold

    # Replace outliers with the nearest non-outlier value (Winsorization)
    lower_bound = X_train[column][~outliers].min()
    List1lower.append(lower_bound)
    upper_bound = X_train[column][~outliers].max()
    List1upper.append(upper_bound)
    X_train.loc[outliers, column] = np.where(X_train.loc[outliers, column] < lower_bound, lower_bound, X_train.loc[outliers, column])
    X_train.loc[outliers, column] = np.where(X_train.loc[outliers, column] > upper_bound, upper_bound, X_train.loc[outliers, column])


# In[49]:


d5=X_train['square_feet'].mean()


# In[50]:


X_train.describe() ###after removing outliers


# IV.Columns dropped

# In[51]:


###Columns to drop : category, fee , currency , price_type  (as they have only one value) 
##Then drop them
###We can drop title, body as they dont affect the data
###column price might be dropped as we already preprocessed price_display so it is a duplicated column
columns_to_drop = ['id', 'category', 'title', 'body', 'currency', 'fee', 'source', 'time','price','price_type','address']
X_train = X_train.drop(columns=columns_to_drop)


# In[52]:


X_train.head(5)


# **V.Feature Engineering**

# In[53]:


##############New Feature ALERTTTTT############ AvgAreaPrice
X_train['longitude_range'] = pd.cut(X_train['longitude'], bins=10)
X_train['latitude_range'] = pd.cut(X_train['latitude'], bins=10)


range_means = X_train.groupby(['longitude_range', 'latitude_range'])['extracted_price'].mean()


X_train['AvgAreaPrice'] = X_train.groupby(['longitude_range', 'latitude_range'])['extracted_price'].transform('mean')


X_train.drop(['longitude_range', 'latitude_range'], axis=1, inplace=True)


# In[54]:


#########NEW FEATURE ALERT!!!!!!!!!!!!################
X_train['NoOfRooms']=X_train['bathrooms']+X_train['bedrooms']


# In[55]:


#########NEW FEATURE ALERT!!!!!!!!!!!!################
X_train['CityStateAvg']=(X_train['CityavgRentprice']+X_train['StateavgRentprice'])/2


# In[56]:


#########NEW FEATURE ALERT!!!!!!!!!!!!################
bath_encoding = X_train.groupby('bathrooms')['extracted_price'].mean().reset_index()
bath_encoding.columns = ['bathrooms', 'avgBathroomsPrice']
X_train = pd.merge(X_train, bath_encoding, on='bathrooms', how='left')


# In[57]:


#########NEW FEATURE ALERT!!!!!!!!!!!!################
X_train['Trial'] = X_train['CityavgRentprice'] + X_train['square_feet']/2 + X_train['avgBathroomsPrice']/3 +X_train['bedrooms']*20


# In[58]:


X_train


# In[59]:


y_train


# In[60]:


y_train.reset_index(drop=True, inplace=True)


# In[61]:


y_train


# 4.Visualizations after preprocessing the features

# In[62]:


merged_train = pd.concat([X_train, y_train], axis=1)
correlation_values = merged_train.corr()['price_display'].sort_values(ascending=False)
print(correlation_values)


# In[63]:


numerical_columns = ['CityavgRentprice','StateavgRentprice','CityStateAvg','AvgAreaPrice']
list2lower =[]
list2upper =[] 
for column in numerical_columns:
    
    z_scores = zscore(X_train[column])

    # Define a threshold for outliers (e.g., Z-score > 3 or < -3)
    threshold = 3

    # Identify outliers
    outliers = np.abs(z_scores) > threshold

    # Replace outliers with the nearest non-outlier value (Winsorization)
    lower_bound = X_train[column][~outliers].min()
    list2lower.append(lower_bound)
    upper_bound = X_train[column][~outliers].max()
    list2upper.append(upper_bound)
    X_train.loc[outliers, column] = np.where(X_train.loc[outliers, column] < lower_bound, lower_bound, X_train.loc[outliers, column])
    X_train.loc[outliers, column] = np.where(X_train.loc[outliers, column] > upper_bound, upper_bound, X_train.loc[outliers, column])


# In[64]:


merged_train = pd.concat([X_train, y_train], axis=1)
correlation_values = merged_train.corr()['price_display'].sort_values(ascending=False)
print(correlation_values)


# In[65]:


#####To see all relations
# Correlation matrix
corr = merged_train.corr()


plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title('Correlation Matrix')
plt.show()


# 5.Feature Selection based on plots, correlations , etc.

# In[66]:


#####getting top features that affect the target to start using them in the model
corr=merged_train.corr()
top_feature = corr.index[corr['price_display'] > 0.32]
print(top_feature)
plt.subplots(figsize=(9, 5))
top_corr = merged_train[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()

Y = pd.DataFrame(y_train['price_display'])
X = X_train[['bathrooms', 'square_feet', 'CityavgRentprice', 'extracted_price',
       'CityStateAvg', 'Trial']]
# X = X.drop(['CityStateAvg'],axis=1)
X ################ Features used in the models


# In[67]:


Y
########Target


# 6.Splitting dataset and perform crossvalidation or train test split

# In[68]:


#############################scaling the features####################
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
s_scaler_path = 'sc_regr.joblib'

dump(sc_X, s_scaler_path)


# In[70]:


merged_test = pd.concat([X_test, y_test], axis=1)


# In[71]:


X_test,y_test=preprocess_data(merged_test)


# In[72]:


X_test


# In[73]:


y_test


# In[74]:


Y = Y.values.reshape(-1, 1)
y_test=y_test.values.reshape(-1, 1)


# 7.Applying Two Different Regression Models or more (Built-in and not Built-in) Performing GD to minimize the error (Extras: Hypertuning)

# **Model I : Linear Regression**

# In[75]:


# Y_np = Y.values.reshape(-1, 1)

cls = linear_model.LinearRegression()

cv_predictions = cross_val_predict(cls, X, Y, cv=5)


mae_lrcv = metrics.mean_absolute_error(Y, cv_predictions)
mse_lrcv = metrics.mean_squared_error(Y, cv_predictions)
rmse_lrcv = np.sqrt(mse_lrcv)
r2_cv = r2_score(Y, cv_predictions)
r2_scorescv = cross_val_score(cls, X, Y, cv=5, scoring='r2')
mean_r2_scores_cv=np.mean(r2_scorescv)

print('Cross-Validated Metrics:')
print('MAE:', mae_lrcv)
print('MSE:', mse_lrcv)
print('RMSE:', rmse_lrcv)
print('R2 Score:', r2_cv)



print('Cross-Validated R2 Scores:', r2_scorescv)
print('Mean R2 Score:',mean_r2_scores_cv )
print('Standard Deviation of R2 Scores:', np.std(r2_scorescv))

cls.fit(X, Y)

r2_training_cv = cls.score(X, Y)
print('Training R2 Score:', r2_training_cv)


# In[76]:


#############with the train test split##########
# Y= Y.values.reshape(-1, 1)
cls = linear_model.LinearRegression()
cls.fit(X,Y)
train_prediction= cls.predict(X)
test_prediction= cls.predict(X_test)
mae_lr = metrics.mean_absolute_error(y_test, test_prediction)
mse_lr =  metrics.mean_squared_error(y_test, test_prediction)
rmse_lr =  np.sqrt(mse_lr)
print('MAE:', mae_lr)
print('MSE:', mse_lr)
print('RMSE:', rmse_lr)
score1 = r2_score(Y,train_prediction)
print("The accuracy of train model is {}%".format(round(score1, 5) *100))
score11 = r2_score(y_test,test_prediction)
print("The accuracy of test model is {}%".format(round(score11, 5) *100))
print('Train Mean Square Error', metrics.mean_squared_error(Y, train_prediction))
print('Test Mean Square Error', metrics.mean_squared_error(y_test, test_prediction))


# In[77]:


plt.scatter(Y, train_prediction, color='blue', label='Training data')

plt.plot(Y, Y, color='red', linewidth=2, label='Regression Line')

plt.title('Actual vs Predicted (Training dataset)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()


plt.scatter(y_test, test_prediction, color='green', label='Test data')

plt.plot(y_test, y_test, color='red', linewidth=2, label='Regression Line')

plt.title('Actual vs Predicted (Test dataset)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()


# **Model II : Ridge Regression**

# In[78]:


# Y= Y.values.reshape(-1, 1)
ridgeReg = Ridge(alpha=10)
ridgeReg.fit(X,Y)
test_predict= ridgeReg.predict(X_test)
train_predict= ridgeReg.predict(X)
mae_ridge = metrics.mean_absolute_error(y_test, test_predict)
mse_ridge =  metrics.mean_squared_error(y_test, test_predict)
rmse_ridge =  np.sqrt(mse_ridge)
print('MAE:', mae_ridge)
print('MSE:', mse_ridge)
print('RMSE:', rmse_ridge)
score5 = r2_score(Y,train_predict)
print("The accuracy of train model is {}%".format(round(score5, 5) *100))
score55 = r2_score(y_test,test_predict)
print("The accuracy of test model is {}%".format(round(score55, 5) *100))
print('Train Mean Square Error', metrics.mean_squared_error(Y, train_predict))
print('Test Mean Square Error', metrics.mean_squared_error(y_test, test_predict))


# In[79]:


plt.scatter(y_test, test_predict, color='blue')
plt.plot(y_test, y_test, color='red', linewidth=2)
plt.title('Actual vs Predicted (Ridge Regression)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()


residuals_ridge = y_test - test_predict

plt.scatter(y_test, residuals_ridge, color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residuals Plot (Ridge Regression)')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.show()


# **Model III : Polynomial Regression**

# In[80]:


poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X)
X_test_poly = poly_features.fit_transform(X_test)
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, Y)
y_train_predicted = poly_model.predict(X_train_poly)
y_test_predicted = poly_model.predict(X_test_poly)
mae_poly = metrics.mean_absolute_error(y_test, y_test_predicted)
mse_poly =  metrics.mean_squared_error(y_test, y_test_predicted)
rmse_poly =  np.sqrt(mse_poly)
print('MAE:', mae_poly)
print('MSE:', mse_poly)
print('RMSE:', rmse_poly)
score6 = r2_score(Y,y_train_predicted)
print("The accuracy of train model is {}%".format(round(score6, 5) *100))
score66 = r2_score(y_test,y_test_predicted)
print("The accuracy of test model is {}%".format(round(score66, 5) *100))
print('Train Mean Square Error', metrics.mean_squared_error(Y, y_train_predicted))
print('Test Mean Square Error', metrics.mean_squared_error(y_test, y_test_predicted))


# In[81]:


plt.scatter(y_test, y_test_predicted, color='blue')
plt.plot(y_test, y_test, color='red', linewidth=2)
plt.title('Actual vs Predicted (Polynomial Regression)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()


# **Model IV : Decision Tree Regressor**

# In[82]:


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor


dt = DecisionTreeRegressor(random_state=42,max_depth=12,min_samples_split=2)
dt.fit(X,Y)
dt_prediction = dt.predict(X_test)
dt_prediction_train = dt.predict(X)


mae_dt = metrics.mean_absolute_error(y_test, dt_prediction)
mse_dt = metrics.mean_squared_error(y_test, dt_prediction)
rmse_dt = np.sqrt(mse_dt)
print('MAE:', mae_dt)
print('MSE:', mse_dt)
print('RMSE:', rmse_dt)


score4 = r2_score(Y, dt_prediction_train)
print("The accuracy of train model is {}%".format(round(score4, 5) * 100))
score44 = r2_score(y_test, dt_prediction)
print("The accuracy of test model is {}%".format(round(score44, 5) * 100))
print('Train Mean Square Error', metrics.mean_squared_error(Y, dt_prediction_train))
print('Test Mean Square Error', metrics.mean_squared_error(y_test, dt_prediction))


# In[83]:


plt.scatter(y_test, dt_prediction, color='blue')
plt.plot(y_test, y_test, color='red', linewidth=2)
plt.title('Actual vs Predicted (Decision Tree)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()


residuals_dt = y_test - dt_prediction.reshape(-1,1)


plt.scatter(y_test, residuals_dt, color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residuals Plot (Decision Tree)')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.show()


# **Model V : Gradient Boosting Regressor**

# In[84]:


# param_grid = {
#     'n_estimators': [50, 100, 150],
#     'max_depth': [2, 3, 4],
#     'learning_rate': [0.01, 0.1, 0.5],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }


GB_reg = GradientBoostingRegressor(n_estimators=300,random_state=42,min_samples_split=5)


GB_reg.fit(X, Y)


test_predict_gb = GB_reg.predict(X_test)
train_predict_gb = GB_reg.predict(X)


mae_GB = metrics.mean_absolute_error(y_test, test_predict_gb)
mse_GB = metrics.mean_squared_error(y_test, test_predict_gb)
rmse_GB = np.sqrt(mse_GB)
print('MAE:', mae_GB)
print('MSE:', mse_GB)
print('RMSE:', rmse_GB)


score7 = r2_score(Y, train_predict_gb)
print("The accuracy of train model is {}%".format(round(score7, 5) * 100))
score77 = r2_score(y_test, test_predict_gb)
print("The accuracy of test model is {}%".format(round(score77, 5) * 100))
print('Train Mean Square Error', metrics.mean_squared_error(Y, train_predict_gb))
print('Test Mean Square Error', metrics.mean_squared_error(y_test, test_predict_gb))


# In[85]:


plt.scatter(y_test, test_predict, color='blue')
plt.plot(y_test, y_test, color='red', linewidth=2)
plt.title('Actual vs Predicted (Gradient Boosting)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()


# **Model VII :Random Forest**

# In[86]:


rf = RandomForestRegressor(random_state=42,n_estimators=250,max_depth=10)

rf.fit(X, Y)

rf_prediction = rf.predict(X_test)
train_predict = rf.predict(X)


mae_rf = metrics.mean_absolute_error(y_test, rf_prediction)
mse_rf = metrics.mean_squared_error(y_test, rf_prediction)
rmse_rf = np.sqrt(mse_rf)
print('MAE:', mae_rf)
print('MSE:', mse_rf)
print('RMSE:', rmse_rf)


score2 = r2_score(Y, train_predict)
print("The accuracy of train model is {}%".format(round(score2, 5) * 100))
score22 = r2_score(y_test, rf_prediction)
print("The accuracy of test model is {}%".format(round(score22, 5) * 100))
print('Train Mean Square Error', metrics.mean_squared_error(Y, train_predict))
print('Test Mean Square Error', metrics.mean_squared_error(y_test, rf_prediction))


# In[87]:


plt.scatter(y_test, rf_prediction, color='blue')
plt.plot(y_test, y_test, color='red', linewidth=2)
plt.title('Actual vs Predicted (Random Forest)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()


residuals = y_test - rf_prediction.reshape(-1, 1)


plt.scatter(y_test, residuals, color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residuals Plot (Random Forest)')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.show()


# **Model VIII: XGBoost**

# In[88]:


xgb_model = xgb.XGBRegressor(n_estimators=50)


xgb_model.fit(X, Y)



y_train_predicted_xgb = xgb_model.predict(X)
y_test_predicted_xgb = xgb_model.predict(X_test)

mae_xgb = metrics.mean_absolute_error(y_test, y_test_predicted_xgb)
mse_xgb = metrics.mean_squared_error(y_test, y_test_predicted_xgb)
rmse_xgb = np.sqrt(mse_xgb)

score_train_xgb = r2_score(Y, y_train_predicted_xgb)
score_test_xgb = r2_score(y_test, y_test_predicted_xgb)

print('MAE (XGBoost):', mae_xgb)
print('MSE (XGBoost):', mse_xgb)
print('RMSE (XGBoost):', rmse_xgb)
score8 = r2_score(Y, y_train_predicted_xgb)
print("The accuracy of train model is {}%".format(round(score8, 5) * 100))
score88 = r2_score(y_test, y_test_predicted_xgb)
print("The accuracy of test model is {}%".format(round(score88, 5) * 100))

print('Train Mean Square Error', metrics.mean_squared_error(Y, y_train_predicted_xgb))
print('Test Mean Square Error', metrics.mean_squared_error(y_test, y_test_predicted_xgb))


# In[89]:


plt.scatter(y_test, y_test_predicted_xgb, color='blue')
plt.plot(y_test, y_test, color='red', linewidth=2)
plt.title('Actual vs Predicted (XGBoost)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()


# **SAVING MODELS**

# In[90]:


##################Saving#################
############Saving the models#########
rf_model_path = 'rf_regr.joblib'
dt_model_path = 'dt_regr.joblib'
gb_model_path = 'gb_regr.joblib'
poly_model_path='poly_regr.joblib'
xgb_model_path='xgb_regr.joblib'


dump(rf, rf_model_path)
dump(dt, dt_model_path)
dump(GB_reg, gb_model_path)
dump(poly_model, poly_model_path)
dump(xgb_model, xgb_model_path)


print("Models saved successfully!")


# In[91]:


rf = load('rf_regr.joblib')
dt= load('dt_regr.joblib')
GB_reg = load('gb_regr.joblib')
poly_model = load('poly_regr.joblib')
xgb_model = load('xgb_regr.joblib')
print("Models loaded successfully!")


# 8.Evaluating the models different results (Accuarcy) also explaining the differences.

# In[92]:


evaluation_metrics = {
    'Model': ['Linear Regression','EnhancedLRwithCV', 'Ridge Regression', 'Polynomial Regression', 'Gradient Boosting','Random Forest', 'Decision Tree','XGBoost'],
    'MAE': [mae_lr,mae_lrcv, mae_ridge, mae_poly, mae_GB, mae_rf, mae_dt,mae_xgb],
    'MSE': [mse_lr,mse_lrcv, mse_ridge, mse_poly, mse_GB, mse_rf, mse_dt,mse_xgb],
    'RMSE': [rmse_lr, rmse_lrcv,rmse_ridge, rmse_poly, rmse_GB,  rmse_rf, rmse_dt,rmse_xgb],
    'R-squared (Training)': [score1,r2_training_cv, score5, score6, score7, score2, score4,score8],
    'R-squared (Testing)': [score11,mean_r2_scores_cv, score55, score66, score77, score22, score44,score88]
}

dff = pd.DataFrame(evaluation_metrics)
dff.set_index('Model', inplace=True)

dff


# 9.Visualizing the models with different plots

# In[93]:


fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].scatter(y_test, test_predict, color='blue')
axes[0].plot(y_test, y_test, color='red', linewidth=2)
axes[0].set_title('Actual vs Predicted (Gradient Boosting)')
axes[0].set_xlabel('Actual Values')
axes[0].set_ylabel('Predicted Values')

axes[1].scatter(y_test, y_test_predicted_xgb, color='blue')
axes[1].plot(y_test, y_test, color='red', linewidth=2)
axes[1].set_title('Actual vs Predicted (XGBoost)')
axes[1].set_xlabel('Actual Values')
axes[1].set_ylabel('Predicted Values')

axes[2].scatter(y_test, rf_prediction, color='blue')
axes[2].plot(y_test, y_test, color='red', linewidth=2)
axes[2].set_title('Actual vs Predicted (Random Forest)')
axes[2].set_xlabel('Actual Values')
axes[2].set_ylabel('Predicted Values')

axes[3].scatter(y_test, dt_prediction, color='blue')
axes[3].plot(y_test, y_test, color='red', linewidth=2)
axes[3].set_title('Actual vs Predicted (Decision Tree)')
axes[3].set_xlabel('Actual Values')
axes[3].set_ylabel('Predicted Values')


plt.tight_layout()


plt.show()


# In[94]:


fig, axes = plt.subplots(1, 4, figsize=(20, 5))


axes[0].scatter(y_train, train_prediction, color='blue', label='Training data')
axes[0].plot(y_train, y_train, color='red', linewidth=2, label='Regression Line')
axes[0].set_title('Actual vs Predicted (Linear Regression, Training dataset)')
axes[0].set_xlabel('Actual Values')
axes[0].set_ylabel('Predicted Values')
axes[0].legend()


axes[1].scatter(y_test, test_prediction, color='green', label='Test data')
axes[1].plot(y_test, y_test, color='red', linewidth=2, label='Regression Line')
axes[1].set_title('Actual vs Predicted (Linear Regression, Test dataset)')
axes[1].set_xlabel('Actual Values')
axes[1].set_ylabel('Predicted Values')
axes[1].legend()


axes[2].scatter(y_test, test_predict, color='blue')
axes[2].plot(y_test, y_test, color='red', linewidth=2)
axes[2].set_title('Actual vs Predicted (Ridge Regression)')
axes[2].set_xlabel('Actual Values')
axes[2].set_ylabel('Predicted Values')


axes[3].scatter(y_test, y_test_predicted, color='blue')
axes[3].plot(y_test, y_test, color='red', linewidth=2)
axes[3].set_title('Actual vs Predicted (Polynomial Regression)')
axes[3].set_xlabel('Actual Values')
axes[3].set_ylabel('Predicted Values')


plt.tight_layout()


plt.show()


# All libraries used versions

# In[95]:


library_versions = [
    ("pandas", pd.__version__),
    ("numpy", np.__version__),
    ("seaborn", sns.__version__),
    ("matplotlib", plt.matplotlib.__version__),
    ("scikit-learn", sklearn.__version__),
    ("mpl_toolkits.mplot3d", "N/A"),  # Axes3D is part of matplotlib, no separate version
    ("itertools", "Built-in"),        # itertools is a built-in module, no version
    ("math", "Built-in"),             # math is a built-in module, no version
    ("LabelEncoder", "Built-in"),     # LabelEncoder is part of scikit-learn, no separate version
    ("linear_model", sklearn.__version__), 
    ("metrics", sklearn.__version__), 
    ("train_test_split", sklearn.__version__), 
    ("ExtraTreesClassifier", sklearn.__version__), 
    ("PolynomialFeatures", sklearn.__version__), 
    ("Ridge", sklearn.__version__), 
    ("DecisionTreeRegressor", sklearn.__version__), 
    ("GradientBoostingRegressor", sklearn.__version__), 
    ("RandomForestRegressor", sklearn.__version__), 
    ("r2_score", sklearn.__version__), 
    ("StandardScaler", sklearn.__version__), 
    ("XGBoost", xgb.__version__)
]

for lib, version in library_versions:
    print(f"{lib} version: {version}")


# In[96]:


####################### THE Script############################
################################THE SCRIPT#######################################
def encode_with_unknown(label_encoder, column):
    # Replace unseen categories with -1
    encoded = column.apply(lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1)
    return encoded
def preprocess_data(new):
    new = new.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    
    # Fill null values
    new['amenities'] = new['amenities'].fillna(0)
    new['square_feet'] = new['square_feet'].fillna(d5)
    new['bedrooms'] = new['bedrooms'].fillna(X_train['bedrooms'].mode().iloc[0])
    new['bathrooms'] = new['bathrooms'].fillna(X_train['bathrooms'].mode().iloc[0])
    new['pets_allowed'] = new['pets_allowed'].fillna('No').apply(lambda x: 'Yes' if x != 'No' else x)
    new['address'] = new['address'].fillna("Unknown")
    new['cityname'] = new['cityname'].fillna(d1)
    new['state'] = new['state'].fillna(d2)
    new['latitude'] = new['latitude'].fillna(X_train['latitude'].mean())
    new['longitude'] = new['longitude'].fillna(X_train['longitude'].mean())
    new['price_type'] = new['price_type'].replace({'Weekly': 'Monthly', 'Monthly|Weekly': 'Monthly'})
    new['category'] = new['category'].replace({'housing/rent/short_term': 'housing/rent/apartment', 'housing/rent/home': 'housing/rent/apartment'})
    new['bedrooms']=new['bedrooms'].replace(0,1)

    
    # Extract price from body column
    new['extracted_price'] = new['body'].apply(extract_price_df)
    new['extracted_price'] = new['extracted_price'].astype(np.float64)
    
    # Encode amenities
    if new['amenities'].dtype == object:
        new['amenities'] = new['amenities'].apply(encode_amenities)
    
    # Label encoding
    label_encoder = load('le.joblib')  # Load label encoder
    new['source'] = encode_with_unknown(label_encoder, new['source'])
    new['price_type'] = encode_with_unknown(label_encoder, new['price_type'])
    new['address'] = encode_with_unknown(label_encoder, new['address'])
    new['body'] = encode_with_unknown(label_encoder, new['body'])
    new['title'] = encode_with_unknown(label_encoder, new['title'])
    new['pets_allowed'] = encode_with_unknown(label_encoder, new['pets_allowed'])
    new['has_photo'] = encode_with_unknown(label_encoder, new['has_photo'])
    new['fee'] = encode_with_unknown(label_encoder, new['fee'])
    new['currency'] = encode_with_unknown(label_encoder, new['currency'])
    
    
    # Drop unnecessary columns
    columns_to_drop = ['id', 'category', 'title', 'body', 'currency', 'fee', 'source', 'time','price_type','address']
    new = new.drop(columns=columns_to_drop)

    # Add new features
    new['NoOfRooms'] = new['bathrooms']+new['bedrooms']
    
    
    city_encoding = X_train.groupby(d4)['extracted_price'].mean().reset_index()
    city_encoding.columns = ['cityname', 'city_price_mean']
    new = pd.merge(new, city_encoding, on='cityname', how='left')
    new['cityname']=new['city_price_mean']
    new.drop(['city_price_mean'], axis=1, inplace=True)
    new['cityname']=new['cityname'].fillna(mostcity)

    #grouping the state according to its names and replacing with the avg price of each state
    state_encoding = X_train.groupby(d3)['extracted_price'].mean().reset_index()
    state_encoding.columns = ['state', 'state_price_mean']
    new = pd.merge(new, state_encoding, on='state', how='left')
    new['state']=new['state_price_mean']
    new.drop(['state_price_mean'], axis=1, inplace=True)
    new['state'] =new['state'].fillna(moststate)
    new = new.rename(columns={'cityname': 'CityavgRentprice', 'state': 'StateavgRentprice'})
    #new['AvgAreaPrice'] = X_train.groupby(['longitude', 'latitude'])['extracted_price'].transform('mean')
    ##############New Feature ALERTTTTT############ AvgAreaState
    X_train['longitude_range'] = pd.cut(X_train['longitude'], bins=10)
    X_train['latitude_range'] = pd.cut(X_train['latitude'], bins=10)

    
    range_means = X_train.groupby(['longitude_range', 'latitude_range'])['extracted_price'].mean()


    new['AvgAreaPrice'] = X_train.groupby(['longitude_range', 'latitude_range'])['extracted_price'].transform('mean')
    
    X_train.drop(['longitude_range', 'latitude_range'], axis=1, inplace=True)
    new['CityStateAvg'] = (new['CityavgRentprice'] + new['StateavgRentprice']) / 2
    bath_encoding = X_train.groupby('bathrooms')['extracted_price'].mean().reset_index()
    bath_encoding.columns = ['bathrooms', 'avgBathroomsPrice']
    new = pd.merge(new, bath_encoding, on='bathrooms', how='left')
    new["avgBathroomsPrice"]= new["avgBathroomsPrice"].fillna(mostBath)

    new['Trial'] = new['CityavgRentprice'] + new['square_feet'] / 2 + new['avgBathroomsPrice'] / 3 + new['bedrooms'] * 20

    X = new[['bathrooms', 'square_feet', 'CityavgRentprice', 'extracted_price',
       'CityStateAvg', 'Trial']]
    new['price_display'] = new['price_display'].str.replace(r'[^0-9.]', '', regex=True).astype(np.int64)
    y=new['price_display']
    y.reset_index(drop=True, inplace=True)
    # Scale features
    scaler = load('sc_regr.joblib')
    X_scaled = scaler.transform(X)
    
    return X_scaled,y


def test_model(dl):
    X, y = preprocess_data(dl)  
    model1 = load('rf_regr.joblib')  
    y_pred1 = model1.predict(X)
    model2 = load('xgb_regr.joblib')  
    y_pred2 = model2.predict(X)
    model3 = load('gb_regr.joblib') 
    y_pred3 = model3.predict(X)

    score1 = r2_score(y,y_pred1)
    print("actual",y)
    print("predicted",y_pred1)
    print("The accuracy of test model 1 RF is {}%".format(round(score1, 5) *100))
    print('Test Mean Square Error for model 1 RF', metrics.mean_squared_error(y, y_pred1))
    score2 = r2_score(y,y_pred2)
    print("The accuracy of test model 2 XGB is {}%".format(round(score2, 5) *100))
    print('Test Mean Square Error for model 2 XGB', metrics.mean_squared_error(y, y_pred2))
    score3 = r2_score(y,y_pred3)
    print("The accuracy of test model 3 GB is {}%".format(round(score3, 5) *100))
    print('Test Mean Square Error for model 3 GB', metrics.mean_squared_error(y, y_pred3))

    return X,y



# In[97]:


#################NEWW DATA HERE##############
new_df = pd.read_csv("ms1trythef.csv")
x,y=test_model(new_df)


# In[ ]:




