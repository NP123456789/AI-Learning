#Importing Libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Objects
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#Importing the Dataset
dataset = pd.read_csv('Machine Learning Practical Application\Data Processing\Data.csv')
#Independent Values also called the matrix of features. The values we want to work out the Dependent value
IndependentValues = dataset.iloc[:, :-1].values
DependentValues = dataset.iloc[:, -1].values

print('##########################################')
print(IndependentValues)
print(DependentValues)


#Replacing the missing data in the csv by the mean value of the rest
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')

#Find what data we want replaced
imputer.fit(IndependentValues[:,1:3])

#Apply the changes to the data 
IndependentValues[:,1:3] = imputer.transform(IndependentValues[:,1:3])


print('##########################################')
print(IndependentValues)

#Encoding the categorical data

#Encoding the independent Variables (Country)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
IndependentValues = np.array(ct.fit_transform(IndependentValues))

print('##########################################')
print(IndependentValues)

#Encoding the dependent variable. Yes and No into 1,0
le = LabelEncoder()
DependentValues = le.fit_transform(DependentValues)

print('##########################################')
print(DependentValues)

#Splitting the data into training and test sets
IndependentValues_train, IndependentValues_test, DependentValues_train, DependentValues_test = train_test_split(IndependentValues, DependentValues, test_size= 0.2, random_state = 1)

print('##########################################')
print('IndependentValues_train')
print(IndependentValues_train)
print('IndependentValues_test')
print(IndependentValues_test)
print('DependentValues_train')
print(DependentValues_train)
print('DependentValues_test')
print(DependentValues_test)

#Feature Scaling
sc = StandardScaler()

IndependentValues_train[: , 3:] = sc.fit_transform(IndependentValues_train[: , 3:])
IndependentValues_test[: , 3:] = sc.transform(IndependentValues_test[: , 3:])


print('##########################################')
print('IndependentValues_train Scaled')
print(IndependentValues_train)
print('IndependentValues_test Scaled')
print(IndependentValues_test)