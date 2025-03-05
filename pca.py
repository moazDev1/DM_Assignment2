# -------------------------------------------------------------------------
# AUTHOR: Moaz Ali
# FILENAME: pca.py
# SPECIFICATION: Calculate PCA
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: 30 mins
# -----------------------------------------------------------*/

#importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


#Load the data
#--> add your Python code here

df = pd.read_csv('heart_disease_dataset.csv')

#Create a training matrix without the target variable (Heart Diseas)
#--> add your Python code here

df_features = df

# Standardize the data

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_features)

#Get the number of features
#--> add your Python code here

num_features = df_features.shape[1]

# Run PCA for 9 features, removing one feature at each iteration

pc1_variances = []
removed_features = []

for i in range(num_features):
    # Create a new dataset by dropping the i-th feature
    # --> add your Python code here
    
    reduced_data = np.delete(scaled_data, i, axis=1)

    # Run PCA on the reduced dataset
    
    pca = PCA()
    pca.fit(reduced_data)

    #Store PC1 variance and the feature removed
    #Use pca.explained_variance_ratio_[0] and df_features.columns[i] for that
    # --> add your Python code here

    pc1_variance = pca.explained_variance_ratio_[0]
    pc1_variances.append(pc1_variance)
    removed_features.append(df_features.columns[i])


# Find the maximum PC1 variance
# --> add your Python code here

max_pc1_variance = max(pc1_variances)
feature_removed = removed_features[pc1_variances.index(max_pc1_variance)]


#Print results
#Use the format: Highest PC1 variance found: ? when removing ?

print(f"Highest PC1 variance found: {max_pc1_variance:.4f} when removing {feature_removed}")