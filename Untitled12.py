#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install klib


# In[1]:


import klib
print(klib.__version__)
import pandas as pd
from zipfile import ZipFile
# %load_ext memory_profiler


# In[2]:


filenames = ["creditcardfraud", "uspollution", "515k-hotel-reviews-data-in-europe"]
for filename in filenames:
    with ZipFile(filename + ".zip", 'r') as zipObj:
        zipObj.extractall('datasets')  


# In[3]:


f_fraud = pd.read_csv("./datasets/creditcard.csv")
df_pollution = pd.read_csv("./datasets/pollution_us_2000_2016.csv")
df_hotel = pd.read_csv("./datasets/Hotel_Reviews.csv")


# In[4]:


f_fraud.head()


# In[6]:


# Cleaning all these dataframes might take a moment
df_fraud_cleaned = klib.data_cleaning(f_fraud, show=None)
print("1/3")
df_pollution_cleaned = klib.data_cleaning(df_pollution, show=None)
print("2/3")
df_hotel_cleaned = klib.data_cleaning(df_hotel, show=None)
print("3/3")


# In[7]:


def byte_to_mb(df):
    return round(df.memory_usage(deep=True).sum()/1024**2,2)


# In[8]:


df_dict = {"Fraud" : f_fraud, "Pollution" : df_pollution, "Hotel" : df_hotel}
for name, df in df_dict.items():
    print(f"{name}".ljust(15), f"Shape: {df.shape}".ljust(25), f"Memory: {byte_to_mb(df)}")


# In[9]:


df_cleaned_dict = {"Fraud" : df_fraud_cleaned, "Pollution" : df_pollution_cleaned, "Hotel" : df_hotel_cleaned}
for name, df in df_cleaned_dict.items():
    print(f"{name}".ljust(15), f"Shape: {df.shape}".ljust(25), f"Memory: {byte_to_mb(df)}")


# In[10]:


def compare_dtypes(df, df2):
    df_dtypes = df.dtypes.rename("dtypes").to_frame().reset_index()
    df_dtypes_cleaned = df2.dtypes.rename("dtypes_cleaned").to_frame().reset_index()
    df_dtypes["index"] = klib.clean_column_names(df).columns
    df_dtypes = df_dtypes.merge(df_dtypes_cleaned, on="index", how="outer").set_index("index").fillna("- dropped -")
    return df_dtypes


# In[29]:


df_fraud_cleaned.columns


# In[12]:


dtypes_fraud = compare_dtypes(f_fraud, df_fraud_cleaned)
dtypes_pollution = compare_dtypes(df_pollution, df_pollution_cleaned)
dtypes_hotel = compare_dtypes(df_hotel, df_hotel_cleaned)


# In[14]:


dtypes_pollution


# In[20]:


print("value_counts():")
get_ipython().run_line_magic('timeit', 'f_fraud.value_counts()')
get_ipython().run_line_magic('timeit', 'df_fraud_cleaned.value_counts()')
print("max():")
get_ipython().run_line_magic('timeit', 'f_fraud.max()')
get_ipython().run_line_magic('timeit', 'df_fraud_cleaned.max()')
print("\nsort_values():")
get_ipython().run_line_magic('timeit', 'f_fraud.sort_values("v1")')
get_ipython().run_line_magic('timeit', 'df_fraud_cleaned.sort_values("v1")')
print("\nnlargest():")
get_ipython().run_line_magic('timeit', 'f_fraud.nlargest(5, "v1")')
get_ipython().run_line_magic('timeit', 'df_fraud_cleaned.nlargest(5, "v1")')


# In[21]:


get_ipython().run_line_magic('memit', 'df_fraud.value_counts()')
get_ipython().run_line_magic('memit', 'df_fraud_cleaned.value_counts()')
get_ipython().run_line_magic('memit', 'df_pollution.value_counts()')
get_ipython().run_line_magic('memit', 'df_pollution_cleaned.value_counts()')
get_ipython().run_line_magic('memit', 'df_hotel.value_counts()')
get_ipython().run_line_magic('memit', 'df_hotel_cleaned.value_counts()')


# In[25]:


klib.cat_plot(df_hotel_cleaned, figsize=(30,30))


# In[31]:


distplot = klib.dist_plot(df_fraud_cleaned["v1"])


# In[ ]:


klib.missingval_plot(df_pollution, figsize=(15,15))


# In[ ]:




