import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import dummy
from sklearn.preprocessing import StandardScaler

# Get the data
training_df = pd.read_csv('data/train.csv')
testing_df  = pd.read_csv('data/test.csv')

# New dataframes to modify and explore
modified_training_df = training_df.copy()
modified_testing_df = testing_df.copy()

# ======== FILL THE MISSING VALUES WITHIN THE WHOLE DATAFRAME ========

# fill the na values withing the Cabin column of the dataframe with Missing
modified_training_df['Cabin'] = modified_training_df['Cabin'].fillna('Missing')
modified_testing_df['Cabin'] = modified_testing_df['Cabin'].fillna('Missing')

# fill the na values within the Embarked column of the dataframe with the mode of the column
modified_training_df['Embarked'] = modified_training_df['Embarked'].fillna(modified_training_df['Embarked'].mode()[0])
modified_testing_df['Embarked'] = modified_testing_df['Embarked'].fillna(modified_testing_df['Embarked'].mode()[0])

# fill the na values within the Age column of the dataframe with the mean
# this is because the age distribution within the age column is quite symmetrical
modified_training_df['Age'] = modified_training_df['Age'].fillna(modified_training_df['Age'].mean())
modified_testing_df['Age'] = modified_testing_df['Age'].fillna(modified_testing_df['Age'].mean())

# and fill the na values within the Fare column of the dataframe with the median
# this is because the distribution of the fare column is quite skewed
modified_training_df['Fare'] = modified_training_df['Fare'].fillna(modified_training_df['Fare'].median())
modified_testing_df['Fare'] = modified_testing_df['Fare'].fillna(modified_testing_df['Fare'].median())

# ======== FILL THE MISSING VALUES WITHIN THE WHOLE DATAFRAME ========





# sort the ages within the Age column of the dataframe into age bins labeled child, adult, and elderly, with the ranges 0-18, 18-65, and 65-100
modified_training_df['AgeBin'] = pd.cut(x=modified_training_df['Age'], bins=[0, 18, 65, 120], labels=['Child', 'Adult', 'Elderly'])
modified_testing_df['AgeBin'] = pd.cut(x=modified_training_df['Age'], bins=[0, 18, 65, 120], labels=['Child', 'Adult', 'Elderly'])

# the ship location is the first letter of the cabin number
modified_training_df['ship_location'] = modified_training_df['Cabin'].astype(str).str[0]
modified_testing_df['ship_location'] = modified_testing_df['Cabin'].astype(str).str[0]

# make a family_size column by adding the SibSp column and the Parch column
modified_training_df['family_size'] = modified_training_df['SibSp'] + modified_training_df['Parch']
modified_testing_df['family_size'] = modified_testing_df['SibSp'] + modified_testing_df['Parch']

# create new dataframes that will be used by the model
model_training_df = modified_training_df.copy()
model_testing_df = modified_testing_df.copy()

# These are the columns that we don't need anymore
drop_list = ['Name', 'Ticket', 'Cabin']
model_training_df.drop(columns=drop_list, inplace=True)
model_testing_df.drop(columns=drop_list, inplace=True)

# These are the categorical columns that we need to encode
dummy_list = ['Pclass', 'Sex', 'Embarked', 'AgeBin', 'ship_location']
model_training_df = pd.get_dummies(data=model_training_df, columns=dummy_list)
model_testing_df = pd.get_dummies(data=model_testing_df, columns=dummy_list)

# The testing dataframe doesn't have this column, and we need both dataframes to be symmetrical
model_testing_df['ship_location_T'] = 0

model_training_df.drop(columns=['PassengerId'], inplace=True) # Don't need this column




# Standardize the rows that we want so it's easier for the model
standardize_list = ['Age', 'SibSp', 'Parch', 'Fare', 'family_size']

# Scale the train_features
train_features = model_training_df[standardize_list]
train_scaler = StandardScaler().fit(train_features.values)
train_features = train_scaler.transform(train_features.values)
model_training_df[standardize_list] = train_features

# Scale the test_features
test_features = model_testing_df[standardize_list]
test_scaler = StandardScaler().fit(test_features.values)
test_features = test_scaler.transform(test_features.values)
model_testing_df[standardize_list] = test_features

dependent_variable = 'Survived'
# y_train is the the column of the dependent variable
y_train = model_training_df[dependent_variable].copy()
# X_train is all the columns except for the dependent variable
X_train = model_training_df.drop(columns=[dependent_variable], axis=1).copy()

# Create a random forest classifier with the criterion gini with 1000 estimators and a random_state of 42
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(
    criterion='gini',
    n_estimators=1000,
    random_state=42
)

# fitted_model = model.fit(X_train.values, y_train.values)