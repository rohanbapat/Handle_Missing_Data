
# Handling Missing Data for Machine Learning Problems

Missing data has always been the bane of scientists’, and more acutely, statisticians’ lives. Today with the advent of superior computing power, this plague is ruining many a-nights of the Data Scientists of today. We are still in a fetal stage when it comes to collecting accurate date to feed the enormous bowels of the computers today. 

Machine learning, the modus operandi of today’s tech leaders deals in information in volumes and velocities unimaginable just 10 years ago. These learning algorithms have a root in the statistical learning theories of the by gone statistics era. Thus, it’s no surprise that missing values, uncollected, lost or corrupted data, still throw a spanner into the well-oiled computations.  Little and Rubin first attempted to answer this problem in 1976. Yet after years and years of scientific hours devoted to this problem, it seems we’re still nowhere near a solution. 

In this post, we explore and evaluate different approaches to handling such data when encountered during machine learning problems.
First, we attempt a theoretical inquiry into handling missing data using simple omission, age old imputations, and machine learning approaches to predicting missing values. Our second goal is to test our data handling methods using machine learning algorithms. By comparing how accurately the algorithms predict trip durations, we can determine which of our approaches to handling missing data was most suited to our chosen dataset, and to machine learning problems in general. 


## Dataset Used:

Our dataset on taxi rides in New York City was taken from Kaggle. We chose this dataset because (a) it contains 1.5 million observations, a large enough dataset for machine learning algorithms to flex their muscles, and (b) it was a full, pre-cleaned dataset with no missing values. This was ideal for our project, evaluating the efficacy of our approaches to handling these values by comparing the prediction accuracy produced by the machine learning algorithms. 

<img src="NYC-Taxi-Cab-Accidents1.jpg">

## Cleaning the dataset

First off, we separated timestamps into individual variables down to the minute from a standard ISO 8601 format. This form of feature engineering enabled us to improve the accuracy of our machine learning algorithms. For example, hour of the day and day of the month, have separate yet meaningful relationships with trip durations. In some bizarre cases, taking the difference between the pickup time and drop-off time revealed some trips that lasted for several days. Some had no passengers and a few had pickup locations in Antarctica. We removed these outliers and truncated those trips which lasted longer than 6 hours, a comfortable upper bound.

## Missing Values:

We generated missing values in two randomly selected yet important predictor variables: Pickup Hour and Drop-off Longitude, removing 33% and 24% of values in these columns. 


### The missing values were approached in three ways: 
1.	Total omission of rows with missing values
    - Advantages:  Easy to implement; Not computationally intensive 
    - Disadvantages: Possibility of wiping out important observations 


```python
# Approach 1 - Delete rows with missing values
# Pass only the dataframe as argument
# Returns clean dataframe and the % of rows dropped

def approach1_rem_msg(messy_df):
    
    # Drop entire row containing missing values
    clean_df = messy_df.dropna()
    
    # Calculate % of rows dropped
    rows_dropped = 1 - clean_df.shape[0]/messy_df.shape[0]
    return clean_df, rows_dropped
```

### Impute missing values: Here, we impute the missing values with a measure of central tendency — ie., mean, median, or mode.  
    - Advantages: No loss of data points and statistical consistency 
    - Disadvantages: Possibility of skewing data without a strong central tendency



```python
#------------------------------------------------------------------------------------------------------------------------

# Approach 2 - Impute missing values
# The following function imputes the missing values with mean/median/mode according to arguments passed
# User also has to pass as list the names of columns which have missing values 
# Call function  - approach2_impute_metric(<df>,<"mean">/<"median">/<"mode">,[<'missingcolname1'>,<'missingcolname2'])
# Returns cleaned df and list of imputed values for all columns

def approach2_impute_metric(messy_df, metric, colnames):
    clean_df = messy_df.copy()    
    missing_list = []
    
    # Impute mean
    if metric=="mean":
        for col in colnames:
            
            # Caluclate mean value of required column
            imputenum = messy_df[col].mean()
            
            # Calculate number of observations having missing value
            missing_count = messy_df[col].isnull().sum()
            
            # Create a list of imputed missing values
            missing_list.append([imputenum]*missing_count)
            
            # Impute mean in the missing fields
            clean_df[col] = messy_df[col].fillna(imputenum)            

    if metric=="median":
        for col in colnames:
            
            # Caluclate median value of required column
            imputenum = messy_df[col].median()
            
            # Calculate number of observations having missing value
            missing_count = messy_df[col].isnull().sum()  
            
            # Create a list of imputed missing vales
            missing_list.append([imputenum]*missing_count)
            
            # Impute median in the missing fields
            clean_df[col] = messy_df[col].fillna(imputenum)
    
    if metric=="mode":
        for col in colnames:
            
             # Caluclate mode value of required column
            imputenum = messy_df[col].mode()
            
            # Calculate number of observations having missing valu
            missing_count = messy_df[col].isnull().sum
            
            # Get positions of missing values
            missing_pos = clean_df[col].isnull()
            
            # In case of multiple modes, randomly allocate the modes across missing fields
            clean_df.loc[clean_df[col].isnull(),col] = np.random.choice(imputenum)
            
            # Create missing_list
            missing_list.append(clean_df.loc[missing_pos,col].tolist())    
        
    return clean_df, missing_list
```

### Predict missing values: This third approach involves predicting missing values using two machine learning algorithms. The following regression algorithms have been implemented: 
a.Linear Regression: A “low-level” machine learning approach which uses the linear relationships between variables to make predictions.

b.Random Forests Regression - A “high-level” machine learning approach that predicts missing values by building an ensemble of decision trees. Unlike linear regression, random forests are capable of handling non-linear relationships.  
    - Advantages: Increased variance based on other predictor variables improves quality of variable for future machine learning
    - Disadvantages: over fitting of missing data producing incorrect result and bias



```python
# Approach 3 - Predict missing values
# The following function predicts missing values using Linear Regression or Random Forests
# User also has to pass as list the names of columns which have missing values 
# Call function  - approach2_impute_metric(<df>,<"Linear Regression">/<"Random Forests">,[<'missingcolname1'>,<'missingcolname2'])
# Returns cleaned df and list of imputed values for all columns
    
def approach3_predict_msg(messy_df, metric, colnames):
    
    # Create X_df of predictor columns
    X_df = messy_df.drop(colnames, axis = 1)
    
    # Create Y_df of predicted columns
    Y_df = messy_df[colnames]
        
    # Create empty dataframes and list
    Y_pred_df = pd.DataFrame(columns=colnames)
    Y_missing_df = pd.DataFrame(columns=colnames)
    missing_list = []
    
    # Loop through all columns containing missing values
    for col in messy_df[colnames]:
    
        # Number of missing values in the column
        missing_count = messy_df[col].isnull().sum()
        
        # Separate train dataset which does not contain missing values
        messy_df_train = messy_df[~messy_df[col].isnull()]
        
        # Create X and Y within train dataset
        msg_cols_train_df = messy_df_train[col]
        messy_df_train = messy_df_train.drop(colnames, axis = 1)

        # Create test dataset, containing missing values in Y    
        messy_df_test = messy_df[messy_df[col].isnull()]
        
        # Separate X and Y in test dataset
        msg_cols_test_df = messy_df_test[col]
        messy_df_test = messy_df_test.drop(colnames,axis = 1)

        # Copy X_train and Y_train
        Y_train = msg_cols_train_df.copy()
        X_train = messy_df_train.copy()
        
        # Linear Regression model
        if metric == "Linear Regression":
            model = LinearRegression()
            model.fit(X_train,Y_train)
            print("R-squared value is: " + str(model.score(X_train, Y_train)))
          
        # Random Forests regression model
        elif metric == "Random Forests":
            model = RandomForestRegressor(n_estimators = 100 , oob_score = True)
            model.fit(X_train,Y_train) 
            
#             importances = model.feature_importances_
#             indices = np.argsort(importances)
#             features = X_train.columns
            
#             print("Missing values in"+ col)
#             #plt.title('Feature Importances')
#             plt.barh(range(len(indices)), importances[indices], color='b', align='center')
#             plt.yticks(range(len(indices)), features) ## removed [indices]
#             plt.xlabel('Relative Importance')
#             plt.show()
        
        X_test = messy_df_test.copy()
        
        # Predict Y_test values by passing X_test as input to the model
        Y_test = model.predict(X_test)
        
        Y_test_integer = pd.to_numeric(pd.Series(Y_test),downcast='integer')
        
        # Append predicted Y values to known Y values
        Y_complete = Y_train.append(Y_test_integer)
        Y_complete = Y_complete.reset_index(drop = True)
        
        # Update list of missing values
        missing_list.append(Y_test.tolist())
        
        Y_pred_df[col] = Y_complete
        Y_pred_df = Y_pred_df.reset_index(drop = True)
    
    # Create cleaned up dataframe
    clean_df = X_df.join(Y_pred_df)
    
    return clean_df,missing_list
```

## Inference:

The approach to handle missing values for machine learning problems is dependent on multiple factors like the size of dataset, extent of missing values, computational costs, accuracy requirements and many more. Depending on the context, one or more of the approaches we explored above can yield significant gains.  

