# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv("Social_Network_Ads.csv")

x= dataset.iloc[:,0:4].values
y=dataset.iloc[:,4:5].values


#Encoding Categorical Data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
column_trans = make_column_transformer((OneHotEncoder(), [1]), remainder='passthrough')
x = column_trans.fit_transform(x)



#splitting the dataset into train data and test data 
from sklearn.model_selection import train_test_split

#creating variable to store x_train,x_test and y_train,y_test
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler

# assigning the standard scscaler to a variable sc_x
sc_x = StandardScaler()

#fitting and transforming the xtrain and the x test
x_train = sc_x.fit_transform(x_train)

#fitting x_test
x_test = sc_x.transform(x_test)

# how to train a module from decision tree
from  sklearn.tree import DecisionTreeRegressor

edith_module = DecisionTreeRegressor(random_state=1)

edith_module.fit(x_train,y_train)

#making a prediction result
prediction_result = edith_module.predict(x_test)
prediction_result
 

#to show how to avalute it is 
from sklearn.metrics import accuracy_score

#Creating a variable to store our accuracy_score
Score = accuracy_score(prediction_result,y_test)

Score
print(Score*100,'%')

