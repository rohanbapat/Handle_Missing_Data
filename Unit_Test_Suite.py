# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 09:43:02 2017

@author: Abhijith
"""

import pandas as pd
import numpy as np
import random
import unittest

#------------------------------------------------------------------------------------------------------------------------

# Approach 1 - Delete rows with missing values
# Pass only the dataframe as argument
#Function returns a new data frame without with all rows with NaNs dropped

def approach1_rem_msg(messy_df):
    clean_df = messy_df.dropna()
    rows_dropped = 1 - clean_df.shape[0]/messy_df.shape[0]
    return clean_df, rows_dropped


#Unit test to determine if approach1_rem_msg retrun a data set without NaNs
class approach1_rem_msg_test(unittest.TestCase): #Initalize class for unit test
    def test_removal_nan_from_df_approach1(self): #define method to test
        df = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD')) #create random data frame df.shape = [100,4]
        random.seed(123)
        ix1= df['A'].sample(round(df.shape[0]/10)).index # select at random indices in column A
        df.loc[ix1,'A']=np.nan# insert NaN values
        df = approach1_rem_msg(df) #Run method to be tested on df and assign to new data frame
        df1 = df[0] #Separate df from tuple
        self.assertNotIn("NaN",df1["A"])# #check if there are any NaNs in the df post method

#------------------------------------------------------------------------------------------------------------------------

def approach2_impute_metric(messy_df, metric, colnames):
    clean_df = messy_df.copy()    
    missing_list = []
    
    if metric=="mean":
        for col in colnames:
            imputenum = messy_df[col].mean()
            missing_count = messy_df[col].isnull().sum()
            missing_list.append([imputenum]*missing_count)
            clean_df[col] = messy_df[col].fillna(imputenum)            

    if metric=="median":
        for col in colnames:
            imputenum = messy_df[col].median()
            missing_count = messy_df[col].isnull().sum()  
            missing_list.append([imputenum]*missing_count)
            clean_df[col] = messy_df[col].fillna(imputenum)
    
    if metric=="mode":
        for col in colnames:
            imputenum = messy_df[col].mode()
            missing_count = messy_df[col].isnull().sum()
            missing_pos = clean_df[col].isnull()
            clean_df.loc[clean_df[col].isnull(),col] = np.random.choice(imputenum, missing_count)
            missing_list.append(clean_df.loc[missing_pos,col].tolist())    
        
    return clean_df, missing_list

#Unit test to determine if approach2_impute_metric retruns a mean value of the column for NaNs in it.
class approach2_impute_metric_test1(unittest.TestCase):#Initalize class for unit test
    def test_impute_mean(self):#define method to test
        df = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))#create random data frame df.shape = [100,4]
        random.seed(123)
        ix1= df['B'].sample(round(df.shape[0]/10)).index # select at random indices in column B
        df.loc[ix1,'B']=np.nan# insert NaN values
        test_mean = pd.isnull(df).any(1).nonzero()[0]#assign all indices in B that have a NaN value to test_mean
        df2 = approach2_impute_metric(df,"mean","B") #Run method to be tested on df and assign to new data frame
        self.assertTrue(df2.loc[test_mean,"B"].all,df["B"].mean())
# AssertTrue returns a successful test when all of the NaN values in the new df
# have been converted to the mean value of column B with NaNs in the original dataframe 

#Unit test to determine if approach2_impute_metric retruns a median value of the column for NaNs in it.
class approach2_impute_metric_test2(unittest.TestCase):#Initalize class for unit test
    def test_impute_median(self):#define method to test
        df = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))#create random data frame df.shape = [100,4]
        random.seed(123)
        df1= df['B'].sample(round(df.shape[0]/10)).index # select at random indices in column B
        df.loc[df1,'B']=np.nan# insert NaN values
        test_median = pd.isnull(df).any(1).nonzero()[0]#assign all indices in B that have a NaN value to test_median
        df2 = approach2_impute_metric(df,"median","B") #Run method to be tested on df and assign to new data frame
        self.assertTrue(df2.loc[test_median,"B"].all,df["B"].median())
# AssertTrue returns a successful test when all of the NaN values in the new df
# have been converted to the median value of column B with NaNs in the original dataframe 

#Unit test to determine if approach2_impute_metric retruns a mode value of the column for NaNs in it.
class approach2_impute_metric_test3(unittest.TestCase):#Initalize class for unit test
    def test_impute_mode(self):#define method to test
        df = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))#create random data frame df.shape = [100,4]
        random.seed(123)
        df1= df['B'].sample(round(df.shape[0]/10)).index # select at random indices in column B
        df.loc[df1,'B']=np.nan# insert NaN values
        test_mode = pd.isnull(df).any(1).nonzero()[0]#assign all indices in B that have a NaN value to test_mode
        df2 = approach2_impute_metric(df,"mode","B") #Run method to be tested on df and assign to new data frame
        self.assertTrue(df2.loc[test_mode,"B"].all,df["B"].mode())
# AssertTrue returns a successful test when all of the NaN values in the new df
# have been converted to the mode value of column B with NaNs in the original dataframe 

#------------------------------------------------------------------------------------------------------------------------




#Initializes unit test module to run test classes
if __name__ == '__main__':
    unittest.main()             