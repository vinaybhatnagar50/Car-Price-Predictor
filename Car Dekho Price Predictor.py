#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings


# In[2]:


warnings.filterwarnings('ignore')


# # 1. Import Library

# In[3]:


import pandas as pd 


# # 2. Import Data

# In[4]:


car=pd.read_csv('C:\\Users\\ok\\Downloads\\Car_Data_From_CARDEKHO.csv')


# In[5]:


car


# # 3. View data (Display top 5 rows)

# In[6]:


car.head()


# # 4. View data (Display last 5 rows)

# In[7]:


car.tail()


# # 5. Information of data

# In[8]:


car.info()


# # 6. Summary Statistics

# In[9]:


car.describe()


# # 7. Checking of missing values

# In[10]:


car.isnull().sum()


# # 8. Check for Categories

# In[11]:


car.nunique()


# # 9. Data Preprocessing

# In[12]:


car.head(1)


# In[13]:


import datetime


# In[14]:


date_time = datetime.datetime.now()


# In[15]:


car['Age'] = date_time.year - car['year']


# In[16]:


car.head()


# # 10. Outlier Removal

# In[17]:


import seaborn as sns


# In[18]:


sns.boxplot(car['selling_price'])


# In[19]:


sorted(car['selling_price'],reverse=True)


# In[20]:


(car['selling_price']>=5500000) & (car['selling_price']<=8900000)


# In[21]:


car[(car['selling_price']>=5500000) & (car['selling_price']<=8900000)]


# In[22]:


car = car[~(car['selling_price']>=5500000) & (car['selling_price']<=8900000)]


# In[23]:


car.shape


# # 11. Encoding the Categorical Columns

# In[25]:


car['fuel'].unique()


# In[26]:


car['fuel'] = car['fuel'].map({'Petrol':0, 'Diesel':1, 'CNG':2, 'LPG':3, 'Electric':4})


# In[27]:


car['fuel'].unique()


# In[28]:


car['seller_type'].unique()


# In[29]:


car['seller_type'] = car['seller_type'].map({'Individual':0, 'Dealer':1, 'Trustmark Dealer':2})


# In[30]:


car['seller_type'].unique()


# In[31]:


car['transmission'].unique()


# In[32]:


car['transmission']= car['transmission'].map({'Manual':0, 'Automatic':1})


# In[33]:


car['transmission'].unique()


# In[34]:


car['owner'].unique()


# In[35]:


car['owner'] = car['owner'].map({'First Owner':0, 'Second Owner':1,'Third Owner':2, 'Fourth & Above Owner':3, 'Test Drive Car':4})


# In[36]:


car['owner'].unique()


# # 12. Store Feature Matrix in X and Response(Target) in Vector y

# In[37]:


X = car.drop(['name','selling_price'],axis=1)


# In[38]:


y = car['selling_price']


# In[39]:


X


# In[40]:


y


# # 13. Splitting the dataset into the Training Set and Test Set

# In[41]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20,random_state=42)


# # 14. Import the Models

# In[42]:


conda install -c conda-forge xgboost


# In[43]:


import xgboost


# In[44]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


# # 15. Fit the Models

# In[45]:


lr = LinearRegression()
lr.fit(X_train,y_train)

rf = RandomForestRegressor()
rf.fit(X_train,y_train)


gbr = GradientBoostingRegressor()
gbr.fit(X_train,y_train)

dt = DecisionTreeRegressor()
dt.fit(X_train,y_train)


xgb = XGBRegressor()
xgb.fit(X_train,y_train)


# # 16. Prediction on Test Data

# In[46]:


y_pred1 = lr.predict(X_test)
y_pred2 = rf.predict(X_test)
y_pred3 = gbr.predict(X_test)
y_pred4 = dt.predict(X_test)
y_pred5 = xgb.predict(X_test)


# # 17. Evaluating the Algorithm

# In[47]:


from sklearn import metrics


# In[48]:


score1 = metrics.r2_score(y_test,y_pred1)
score2 = metrics.r2_score(y_test,y_pred2)
score3 = metrics.r2_score(y_test,y_pred3)
score4 = metrics.r2_score(y_test,y_pred4)
score5 = metrics.r2_score(y_test,y_pred5)


# In[49]:


print(score1,score2,score3,score4,score5)


# In[50]:


final_data =pd.DataFrame({'Models':['LR','RF','GBR', 'dt','XGB'],
              "R2_SCORE":[score1,score2,score3,score4,score5]})


# In[51]:


final_data


# In[52]:


import seaborn as sns
sns.barplot(x=final_data['Models'],y=final_data['R2_SCORE'], alpha=0.8)


# In[53]:


import seaborn as sns
sns.pairplot(car)


# # 18. Save the Model

# In[54]:


xgb = XGBRegressor()
xgb_final = xgb.fit(X_train,y_train)


# In[55]:


import joblib


# In[56]:


joblib.dump(xgb_final,'car_dekho_price_predictor')


# In[57]:


model = joblib.load('car_dekho_price_predictor')


# # 19. Prediction on New Data

# In[58]:


import pandas as pd 
data_new = pd.DataFrame({
   'year':2014,
   'km_driven':70000,
   'fuel':0,
   'seller_type':0,
   'transmission':0,
   'owner':0,
   'Age' :9,
},index=[0])


# In[59]:


model.predict(data_new)

