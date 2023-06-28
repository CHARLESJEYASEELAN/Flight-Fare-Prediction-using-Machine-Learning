import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score

#Loading Data Set - It is a cleaned DS so there are no null values
ds = pd.read_csv("Cleaned_dataset.csv")

#Splitting Input Features and Target Variable i.e., Fare
X,y = ds.drop(columns=['Date_of_journey','Journey_day','Flight_code','Source','Departure','Arrival','Destination']),ds['Fare']

#PERFORMING EDA

#Analysing Categorical Features 
plt.scatter(np.array(X["Airline"]),np.array(y))

plt.scatter(np.array(X["Class"]),np.array(y))

plt.scatter(np.array(X["Total_stops"]),np.array(y))

#Converting Categorical data into numberical Data using label Encoding
le = LabelEncoder()
labelAirline = le.fit_transform(X["Airline"])
labelClass = le.fit_transform(X["Class"])
labelStops = le.fit_transform(X["Total_stops"])
X.drop("Airline",axis=1,inplace=True)
X.drop("Class",axis=1,inplace=True)
X.drop("Total_stops",axis=1,inplace=True)
X["Airline"] = labelAirline
X["Class"] = labelClass
X["Total_stops"] = labelStops

#SPLITING THE DATASET to for Training and Testing
#Test size 33%
X_Train,X_Test,Y_Train,Y_Test = train_test_split(X,y,test_size=0.33,random_state=42)

#Converting the Target variable to non numerical data (if it contains any numerical ones)
Y_Train = Y_Train.astype(int)
Y_Test = Y_Test.astype(int)


#Scaling the data
sc = StandardScaler() 
X_Train = sc.fit_transform(X_Train)
X_Test = sc.fit_transform(X_Test)

#Creating the model and training the model
Reggresor = RandomForestRegressor(n_estimators=100)
Reggresor.fit(X_Train,Y_Train)

#TrainingPrediction and Evaluation 
TrainingPrediction = Reggresor.predict(X_Train)
print('Mean Absolute Error:', mean_absolute_error(Y_Train, TrainingPrediction))
print('Mean Squared Error:', mean_squared_error(Y_Train, TrainingPrediction))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(Y_Train, TrainingPrediction)))


#Testing the MODEL with Test Data set
TestingPrediction = Reggresor.predict(X_Test)

#Evaluation of the model
print('Mean Absolute Error:', mean_absolute_error(Y_Test, TestingPrediction))
print('Mean Squared Error:', mean_squared_error(Y_Test, TestingPrediction))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(Y_Test, TestingPrediction)))
print("r2score: ",r2_score(Y_Test,TestingPrediction))
