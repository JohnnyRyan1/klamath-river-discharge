#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Predict Klamath River discharge using machine learning.

"""

# Import modules
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define filepath
filepath = '/home/johnny/Documents/Teaching/490_Geospatial_Data_Science_Applications/Applications/River_Discharge/data/'

# Import data
training_data = pd.read_csv(filepath + 'era/era5_training_data.csv')
label_data = pd.read_csv(filepath + 'discharge/Klamath_Discharge_Daily_2019.csv')

# Define feature list
feature_list = ['t2m', 'mer', 'mtpr', 'swvl1', 'msmr', 'sd']

# Define labels and targets
y = label_data['discharge']
X = training_data[feature_list]

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classifier
classifier = RandomForestRegressor(n_estimators=100)

# Train classifier
classifier.fit(X_train, y_train)

# Predict
predictions = classifier.predict(X_test)

# Calculate the absolute errors
errors = abs(predictions - y_test)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 3))

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# Get numerical feature importances
importances = list(classifier.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, 
                       importance in zip(feature_list, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

df_preds = pd.DataFrame(list(zip(y_test, predictions)), 
                        columns=['y_test', 'predictions'])

###############################################################################
# Apply model to entire discharge record
###############################################################################

# Make predictions
predictions = classifier.predict(training_data[feature_list])

# Add to original DataFrame
label_data['predictions'] = predictions
                
# Plot
plt.plot(label_data['discharge'])
plt.plot(label_data['predictions'])


plt.scatter(label_data['discharge'], label_data['predictions'])
plt.plot([0,120000],[0,120000])














