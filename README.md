# BDA-HW2-PCA
Tools used:  
-Python  

Problem Statement:  
Use Principal Components Analysis to identify the major components of variation in the ratings amongst cities from the given dataset.  
1. Form a data matrix X whose dimensions are NxP, where N are the number of cities (329) and P are the
number of features (9)  
2. Normalize X by transforming each feature column to zero mean and unit standard deviation.  
3. Perform PCA and calculate explained variance ratios, loading vectors etc. You may use scikit-learn’s PCA functions (sklearn.decomposition.PCA)  
  
Now answer the following questions  
1. Plot explained variance ratio as a function of number of principal components. What are minimum number of principal components needed to explain at least 80% of the variance in the data.  
2. List the loading vectors for the first 3 principal components and interpret them.  
3. Transform the original data into the principal component space. Plot the transformed data in PCA1-PCA2, PCA1-PCA3 and PCA2-PCA3 space, i.e. a biplot like we discussed in class. Plot the attribute axes on the plot (note the plot may get a bit messy if you are using full city names in the plot. )  
4. Discuss the plots. Identify any unusual cities (i.e. outliers in these plots) 
