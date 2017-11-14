#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Titanic Passenger Data

"""

import numpy as np
import pandas as pd
import visuals as vs
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

train = pd.read_csv('titanic_data.csv')
train.head()
#removing Survived feature from DataFrame
outcomes = train['Survived']
data = train.drop('Survived', axis = 1)
display(data.head())

#Out of the first five passengers, if we predict 
#that all of them survived, what would you expect the accuracy of our predictions to be?

def accuracy_score(truth, pred):
    # Ensure that the number of predictions matches number of outcomes
    if len(truth) == len(pred): 
        # Calculate and return the accuracy as a percent
        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean()*100)
    else:
        return "Number of predictions does not match number of outcomes!"
# Test the 'accuracy_score' function
predictions = pd.Series(np.ones(5, dtype = int))
print accuracy_score(outcomes[:5], predictions)


#Making Predictions:
def predictions_0(data):
    predictions = []
    for _, passenger in data.iterrows():
        # Predict the survival of 'passenger'
        predictions.append(0)
    # Return our predictions
    return pd.Series(predictions)
# Make the predictions
predictions = predictions_0(data)


print accuracy_score(outcomes, predictions)
vs.survival_stats(data, outcomes, 'Sex')

def predictions_1(data):
    predictions = []
    for _, passenger in data.iterrows():
        if passenger['Sex'] == 'female':
            predictions.append(1)
        else:
            predictions.append(0)
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_1(data)
print accuracy_score(outcomes, predictions)

vs.survival_stats(data, outcomes, 'Age', ["Sex == 'male'"])


def predictions_2(data):
    predictions = []
    for _, passenger in data.iterrows():       
        if passenger['Sex'] == 'female':
            predictions.append(1)
        else:
            if passenger['Age'] < 10:
                predictions.append(1)
            else:
                predictions.append(0)
    
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_2(data)
print(accuracy_score(outcomes, predictions))


survival_stats(data, outcomes, 'Age', ["Sex == 'male'", "Pclass == 1", "Parch > 0"])
survival_stats(data, outcomes, 'Age', ["Sex == 'male'", "Pclass == 1"])


def predictions_3(data):
    
    predictions = []
    for _, passenger in data.iterrows():
        if passenger['Sex'] == 'female':
            if passenger['Age'] > 40 and passenger['Age'] < 60 and passenger['Pclass'] == 3:
                predictions.append(0)
            else:
                predictions.append(1)
        else:
            if passenger['Age'] <= 10:
                predictions.append(1)
            elif passenger['Pclass'] == 1 and passenger['Age'] <= 40:
                predictions.append(1)
            else:
                predictions.append(0)
  
    
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_3(data)


"""

Logistic Regression

"""

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Pclass', data=train, palette='RdBu_r')


sns.distplot(train['Age'].dropna(),kde=False, bins=30)

train['Age'].plot.hist(bins=35)

train.info()

sns.countplot(x='SibSp', data=train)

train['Fare'].hist(bins=40,figsize=(10,4))

import cufflinks as cf
cf.go_offline()
train['Fare'].iplot(kind='hist',bins=50)

train['Age'].hist(bins=30,color='darkred',alpha=0.7,by=train['Survived']) 

plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass', y='Age', data=train)


#imputation
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else: 
            return 24
    else:
        return Age

#Cleaning Data
train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')

train.drop('Cabin', axis=1, inplace=True)
train.head()

  
train.dropna(inplace=True)


#Categorical Features: dummy variable 
#Multi-collinearity
sex = pd.get_dummies(train['Sex'], drop_first=True)
sex.head()

