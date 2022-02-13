from logging import raiseExceptions
from turtle import right
from typing import Type
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from pyparsing import col

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class FeatureEngineering:
    
    def __init__(self):
        pass

    def describe_df(self, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError
        else:
            print('DataFrame characteristics: \n')
            nulls = df.isnull().sum() #Checking for null values
            if nulls.sum() == 0:
                print('There are no missing values.')
            else:
                print('Null values:\n')
                print(nulls)    
            
            duplicated = df.duplicated() #Checking for duplicated values
            if df[duplicated == True].empty:
                print('\nThere are duplicated values.')
            else:
                print(df[duplicated])
            
            cols = df.columns #Checking numeric and categorical variables
            num_cols = df._get_numeric_data().columns 
            cat_cols = list(set(cols) - set(num_cols))
            print('\nNumeric variables: ', list(num_cols))
            print('\nCategorical variables: ', list(cat_cols))
            
            dtypes = df.dtypes #Checking variable data types
            print('\nThe variables in the dataframe present the following data types: ')
            print(dtypes)
            
            print('\nDataset description: ') #Printing a description of the dataset
            print(df.describe()) 

        return 

    def plot_dist(self, df, features=[], use_classes=True):
        if not isinstance(df, pd.DataFrame) or not isinstance(features, list):
            raise TypeError
        else:    
            if not features: #Checking if the "features" parameters is passed by the user or not
                features = df._get_numeric_data().columns
            
            ngraphs = len(features)
            if ngraphs >= 3:
                ncols = 3
            else:
                ncols = ngraphs

            nrows = math.ceil(ngraphs/ncols)

            col = 0
            row = 0
            
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
            
            df = df[features]
            for column in df.columns:
                if use_classes is True:
                    k = 1 + math.ceil(np.log2(len(df[column])))
                    hist = df[column].plot.hist(ax=axs[row,col], bins=k)
                else:
                    hist = df[column].plot.hist(ax=axs[row,col])
                hist.set(xlabel=column)
                if col == 2:
                    col = 0
                    row += 1 
                else:
                    col += 1

            #Deleting empty plots of the subplot
            nempty_graphs = nrows*ncols - ngraphs
            if nempty_graphs > 0:
                for i in range(0, nempty_graphs):
                    fig.delaxes(axs[nrows-1][ncols-1-i])

            fig.set_size_inches(18.5, 10.5)

        return 
        
    def remove_outliers_by_quantiles(self, df, quantile_size, factor=1.5, features=[]):
        #Removes outliers based on the interquantile range analysis
        #quantile_size is the percentage of the first quantile considered in the analysis 
        if not isinstance(df, pd.DataFrame) or not isinstance(features, list):
            raise TypeError
        else:
            if not features:
                features = df._get_numeric_data().columns
                
            Q1 = df[features].quantile(quantile_size)
            Q2 = df[features].quantile(1 - quantile_size)
                
            IQR = Q2 - Q1
            self.df = df[~((df[features] < (Q1 - factor*IQR)) | (df[features] > (Q2 + factor*IQR)) ).any(axis=1)]

            return self.df
    
    def remove_outliers_by_model(self, df, num_features=[], n_estimators=100, max_samples='auto', contamination='auto'):
        #Removes outliers by using an IsolationForest model
        if not isinstance(df, pd.DataFrame) or not isinstance(num_features, list):
            raise TypeError
        else:    
            if not num_features: #Checking if the "features" parameters is passed by the user or not
                num_features = df._get_numeric_data().columns
            
            cat_features = list(set(df.columns) - set(num_features))
            
            df_num = df[num_features]
            df_cat = df[cat_features]

            df_num = df[num_features]
            model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, contamination=contamination)
            pred = model.fit_predict(df_num)

            #Converts predictions into a pandas Series, adds to the dataframe and the eliminates the outliers
            df_num['pred'] = pd.Series(pred)
            self.df = pd.merge(df_num, df_cat, left_index=True, right_index=True)
            self.df = self.df[self.df['pred'] == 1]
            self.df = self.df.drop(columns=['pred'])

            return self.df

    def encode_features(self, df, method, features=[], target=None):
        if not isinstance(df, pd.DataFrame) or not isinstance(features, list):
                raise TypeError
        else:    
            if not features: #Checking if the "features" parameters is passed by the user or not
                features = list(set(df.columns) - set(df._get_numeric_data()))

        supported_methods = ['one_hot', 'label', 'mean']
        if method not in supported_methods:
            raise ValueError('Unsupported encoding method')

        if method == 'one_hot':
            self.df = pd.get_dummies(df, columns=features)

        elif method == 'label':
            for feature in features:
                df['{}_enc'.format(feature)] = LabelEncoder().fit_transform(df[feature])
            self.df = df.drop(columns=features)

        elif method == 'mean':
            if target == None:
                raise ValueError('target feature not informed')
            else:
                for feature in features:
                    mean_map = df.groupby(feature)[target].mean()
                    df.loc[:, '{}_enc'.format(feature)] = df[feature].map(mean_map)
                self.df = df.drop(columns=features)
        
        return self.df
