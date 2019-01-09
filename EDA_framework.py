# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 15:21:53 2018

@author: DANIEL MARTINEZ
"""

import pandas as pd

def get_missing_data_table(dataframe):
    total = dataframe.isnull().sum()
    percentage = dataframe.isnull().sum() / dataframe.isnull().count()
    
    missing_data = pd.concat([total, percentage], axis='columns', keys=['TOTAL','PERCENTAGE'])
    return missing_data.sort_index(ascending=True)

def get_null_observations(dataframe, column):
    return dataframe[pd.isnull(dataframe[column])]

def delete_null_observations(dataframe, column):
    fixed_df = dataframe.drop(get_null_observations(dataframe,column).index)
    return fixed_df
    
def transform_dummy_variables(dataframe, columns):
    df = dataframe.copy()
    for column in columns:    
        df[column] = pd.Categorical(df[column])
    df = pd.get_dummies(df, drop_first=False)
    return df

def imput_nan_values(dataframe, column, strateg, fill_value=None):
    from sklearn.impute import SimpleImputer
    if strateg == 'constant':
        imp = SimpleImputer(strategy=strateg, fill_value=fill_value)
    else:
        imp = SimpleImputer(strategy=strateg)
    df = dataframe.copy()
    df[column] = imp.fit_transform(df[column].values.reshape(-1,1))
    return df

def delete_column(dataframe, column):
    return dataframe.drop([column], axis='column')

def get_upper_outliers(dataframe, column):
    h_spread = dataframe[column].quantile(.75) - dataframe[column].quantile(.25)
    limit = dataframe[column].quantile(.75) + 2 * h_spread
    return dataframe[dataframe[column] > limit]

def get_lower_outliers(dataframe, column):
    h_spread = dataframe[column].quantile(.75) - dataframe[column].quantile(.25)
    limit = dataframe[column].quantile(.25) - 2 * h_spread
    return dataframe[dataframe[column] < limit]

def delete_outliers(dataframe, column, hinge):
    if hinge == 'upper':
        outliers = get_upper_outliers(dataframe, column)
    elif hinge == 'lower':
        outliers = get_lower_outliers(dataframe, column)
    
    return dataframe.drop(outliers.index)
    