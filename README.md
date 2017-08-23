#Handling Missing Data

##Aim - 

Missing values are a data scientist’s nemesis. Real world machine learning datasets contain missing values, but machine learning algorithms do not accept missing values as input. Our project explores and evaluates different approaches to handling missing data. 
Our project goals fall into two distinct parts. The first is a theoretical inquiry into handling missing data. We explored various methods of handling a dataset with missing values, including machine learning approaches to predicting missing values and validating our predictions. In order to do this, we generated missing values in our dataset and applied various data handling techniques to the artificial missing data.
Our second goal was to test our data handling methods using machine learning algorithms. By comparing how accurately the algorithms were able to predict trip durations, we can determine which of our approaches to handling missing data was most suited to our chosen dataset, and to machine learning problems in general. 

##Project Objectives -  

(1) Explore the original dataset to produce summary statistics on taxi ride data in New York City. 
(2) Explore non-machine learning approaches to handling missing values. 
(3) Explore machine learning approaches to missing data by predicting missing values. 
(4) Use machine learning to test the accuracy of the approaches outlined above. 

##Dataset -

Our dataset on taxi rides in New York City was taken from Kaggle. We chose this dataset because (a) it contains 1.5 million observations, a large enough dataset for machine learning to be applicable, and (b) it was a full, pre-cleaned dataset with no missing values. This was ideal for the purpose of our project so that we could evaluate the efficacy of our approaches to handling these values by comparing the prediction accuracy produ ced by the machine learning algorithms. 

Data Pre-Processing and Exploratory Analysis -

We separated timestamps into individual variables for the hour and minute, and separated dates into individual variables with day, month and year. We did this in order to make these variables easy to analyze using machine learning techniques. 
It was also necessary to remove some outliers. In some bizarre cases, taking the difference between the pickup time and drop-off time revealed some trips that lasted for several days. We removed rides that lasted longer than 6 hours, as well as those that logged 0 passengers.  
                                  
How to handle missing values? 

We generated missing values in two randomly selected variables: Pickup Hour and Drop-off Longitude, removing 33% and 24% of values in these columns. We then handled missing values in three ways. Each approach takes a dataset with missing values as input and outputs dataset with the missing values filled with new values. Our three approaches are: 
1. Remove rows with missing values: In this approach, rows with missing values are simply eliminated. 
- Advantages:  Easy to implement; Not computationally intensive 
- Disadvantages: Possibility of wiping out important observations 
2. Impute missing values: Here, we impute the missing values with a measure of central tendency — ie., mean, median, or mode.  In the case that multiple modes were found, we randomly assigned one of the modes to each missing field. 
- Advantages: No loss of data points and statistical consistency 
- Disadvantages: Possibility of skewing data without a strong central tendency
3. Predict missing values: The third approach involves predicting missing values using machine learning algorithms. The rows containing missing values in the NYC taxi dataset are our test set, while the rows containing non-missing values form the train dataset. The following regression algorithms have been implemented: 
  a. Linear Regression: A “low-level” machine learning approach which uses the linear relationships between variables to make predictions. 
  b. Random Forests Regression - A “high-level” machine learning approach that missing values by building an ensemble of decision trees. Unlike linear regression, random forests are capable of handling non-linear relationships.  
- Advantages: Increased variance based on other predictor variables improves quality of variable for future machine learning
- Disadvantages: over fitting of missing data producing incorrect result and bias

Evaluate approaches to handling missing values-

With the missing values replaced, we can now evaluate the efficacy of our various data handling approaches in the context of machine learning problems. First, we classified the “ride duration” variable into a binomial variable — rides completed in 20 minutes and under took on a value of 0, whilst all other rides took on a value of 1. Then, we used Logistic Regression and Random Forests to classify the binomial ride duration in each of our 6 altered datasets. The accuracy of the classifications is displayed below: 
                           
The above summary shows that for the given dataset, the best results are yielded by removing the rows containing missing values. For datasets containing larger datasets with more missing values, we can opt to use a combination of these approaches to yield the best results. 

Testing -

● The code development followed a TDD where each method was first tested empirically. 
● We tested the functions that had the following behaviors: Dropping the NaNs; imputing a mean, median or mode; and applying linear regression and random forests to predict desired values.
● Each function its own unit test, to localize any failures and develop leaner code.  
● Further, we had built-in interactive features which made the set up complex. This required intensive testing to cover all flow paths. 

Conclusion - 

The approach to handle missing values within machine learning datasets is dependent on multiple factors, including size of dataset, number of missing values, computational costs, accuracy requirements, etc. Depending on the context, one or more of the approaches we explored above can be taken. With some further development, we can turn our program into a data product that data scientists can use to investigate different approaches to handling missing values in their datasets. 
User Interaction We implemented user interactive plots and graphs to aid our analysis using matplotlib and ipywidgets, which produces interactive widgets for Jupyter notebook. 
Advanced Queries We investigated several approaches to handling missing data, including the use of machine learning to predict missing values. In addition, we evaluated the efficacy of our approaches through machine learning algorithms. 
