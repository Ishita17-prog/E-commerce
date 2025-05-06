#!/usr/bin/env python
# coding: utf-8

# In[121]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


# # **E-Commerce Product Performance Dataset**
# 
# **Description**
# This realistic dataset contains 2,000 records representing product performance metrics in an e-commerce environment.
# 
# Columns Explanation:
# 
# * Product_Price: The listed price of the product in USD (range: 5 to 1000).
# * Discount_Rate: Discount rate applied to the product (0.0 to 0.8).
# * Product_Rating: Customer rating on a scale from 1 to 5.
# * Number_of_Reviews: Total number of user reviews (0 to 5000, highly skewed).
# * Stock_Availability: Product availability in stock (1 = available, 0 = out of stock).
# * Days_to_Deliver: Number of days it takes to deliver the product (1 to 30).
# * Return_Rate: Proportion of items returned after purchase (0.0 to 0.9).
# * Category_ID: ID of the product category (integer from 1 to 10).
# 
# 1:'Sports & Outdoors',2:'Beauty & Cosmetics',3:'Toys & Games',4:'Fashion & Apparel',5:'Health & Personal Care',6:'Home & Kitchen',7:'Books & Stationery',8:'Groceries & Gourmet Foods',9:'Electronics',10:'Automotive Accessories'

# In[122]:


df=pd.read_csv('C:\\Users\\ISHITA GUPTA\\OneDrive\\Desktop\\html work\\Pandas\\Data\\Internship\\ecommerce_product_performance.csv')


# In[123]:


df.head()


# In[125]:


df.shape


# In[126]:


df.info()


# In[127]:


df.describe()


# In[128]:


df.isnull().sum()


# <span style='color:red'>Firstly, treating the null values for the **Category_ID** column. To avoid the loss of information I have filled the null values as the mode(the highest occuring category id) </span>

# In[129]:


df['Category_ID'].value_counts()


# In[130]:


df['Category_ID']=df['Category_ID'].fillna(df['Category_ID'].mode()[0])


# In[131]:


df['Category_ID'].isnull().sum()


# In[132]:


df['Category_ID'].value_counts()


# <span style='color:red'>We can see that mode is Groceries & Gourmet Foods  and the value count for category increases from 210 to 310 </span>

# <span style='color:red'> Now fill the null values of other columns by the mean value of that column when grouped category_id wise. </span>

# In[133]:


list=['Product_Price','Discount_Rate','Product_Rating','Number_of_Reviews','Stock_Availability','Days_to_Deliver','Return_Rate']
for i in list:
    df[i]=df.groupby('Category_ID')[i].transform(lambda x: x.fillna(x.mean()))


# In[134]:


df.isnull().sum()


# In[135]:


df.duplicated().sum()


# In[136]:


df[df.duplicated()]


# In[137]:


df.drop_duplicates()


# <span style='color:red'> This data set does not contain any duplicate values. First I checked it and then if there is any duplicate values, we can drop it, we can drop it by subset wise and also we can keep the values as per our choice whether the first occurence or the last one.</span>

# In[138]:


df[['Number_of_Reviews','Stock_Availability','Days_to_Deliver']]=df[['Number_of_Reviews','Stock_Availability','Days_to_Deliver']].astype('int')


# In[139]:


df.info()


# <span style='color:red'> Above these columns have been converted to integers.</span>

# <span style='color:red'> Normalization of numerical columns.</span>

# In[140]:


numeric =['Discount_Rate','Product_Rating','Number_of_Reviews','Stock_Availability','Days_to_Deliver','Return_Rate']
df[numeric] = MinMaxScaler().fit_transform(df[numeric])


# In[141]:


df[numeric_cols].min()


# In[142]:


df[numeric_cols].max()


# In[143]:


df.to_csv('cleaned_data.csv', index=False)


# In[144]:


folder_path ='C:\\Users\\ISHITA GUPTA\\OneDrive\\Desktop\\html work\\Pandas\\Data\\Internship' 

import os
if not os.path.exists('C:\\Users\\ISHITA GUPTA\\OneDrive\\Desktop\\html work\\Pandas\\Data\\Internship'):
    os.makedirs('C:\\Users\\ISHITA GUPTA\\OneDrive\\Desktop\\html work\\Pandas\\Data\\Internship')

df.to_csv(os.path.join('C:\\Users\\ISHITA GUPTA\\OneDrive\\Desktop\\html work\\Pandas\\Data\\Internship', 'cleaned_data.csv'), index=False)


# <span style='color:red'> Used barplot to show the average price for each of the category and also added a dotted line to show the average price for all the products.</span>

# In[145]:


df=pd.read_csv("C:\\Users\\ISHITA GUPTA\\OneDrive\\Desktop\\html work\\Pandas\\Data\\Internship\\cleaned_data.csv")


# In[148]:


avg_price_category=df.groupby('Category_ID')['Product_Price'].mean().reset_index()
avg_price=df['Product_Price'].mean()
plt.figure(figsize=(6,5))
sns.barplot(avg_price_category,x='Category_ID',y='Product_Price',hue='Category_ID',palette='rocket')
plt.axhline(y=avg_price,color='r',linestyle='--',linewidth=2)
plt.title('Category wise average price')
plt.xlabel('Category_ID')
plt.legend().remove()
plt.show()


# <span style='color:red'> We can see that Automobile Accessories has the highest average price and category Books and stationary has the lowest average price.</span>

# In[153]:


plt.figure(figsize=(8,5))
sns.heatmap(df.corr(),annot=True,linewidth=0.5,cmap = 'summer')
plt.title('Correlation Between Features')
plt.show()


# In[156]:


plt.figure(figsize=(6,5))
sns.histplot(df['Return_Rate'], bins=25, kde=True, color='red')
plt.title('Distribution of Product Return Rates')
plt.xlabel('Return Rate')
plt.ylabel('Number of Products')
plt.show()


# It is rightly skewed distribution which shows fewer products with a very high return which is a good sign. It can be also observed that a lot of products get returned about 30% of the time.
