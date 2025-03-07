import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading the csv data to a Pandas DataFrame
credit_card_data = pd.read_csv('creditcard.csv')

#First 5 rows of the dataset
credit_card_data.head()

credit_card_data.tail()

#dataset informaion
credit_card_data.info()

# Check for missing values in each column
credit_card_data.isnull().sum()

#Check the distribution of legit transaction and fraudulent transaction
credit_card_data['Class'].value_counts()

"""This dataset is highly unbalanced

0 is Normal transaction,
1 is Fraudulent transaction
"""

#seperation of the data for analysis
legit=credit_card_data[credit_card_data.Class==0]
fraud=credit_card_data[credit_card_data.Class==1]
print(legit.shape)
print(fraud.shape)

#Statistical measures of the data
legit.Amount.describe()

fraud.Amount.describe()

# Comparing the values for legit and fraud transactions
credit_card_data.groupby('Class').mean()

"""Under-Sampling: Building a sample dataset containing similar distribution of the legit transaction and the fraudulent transaction

The number of fraudulent transactions is 492
"""

legit_sample= legit.sample(n=85)

"""Concatenating the two Dataframes"""

new_dataset=pd.concat([legit_sample,fraud],axis=0)

new_dataset.head()

new_dataset['Class'].value_counts()

new_dataset.groupby('Class').mean()

"""Splitting the data into the features and targets"""

x=new_dataset.drop(columns='Class',axis=1)
y=new_dataset['Class']

print(x)

print(y)

"""Split the data into training data and testing data"""

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)

print(x.shape,X_train.shape,X_test.shape)

"""Train model using Logistiic Regression for Binary classification Model

Model evaluation based on the accuracy score
"""

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Step 1: Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Step 2: Fit the model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_scaled, Y_train)

X_train_prediction = model.predict(X_train_scaled)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print("Accuracy on Training data: ", training_data_accuracy)

X_test_prediction = model.predict(X_test_scaled)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print("Accuracy on Test data: ", test_data_accuracy)
