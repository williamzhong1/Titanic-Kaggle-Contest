import pandas as pd
from sklearn.naive_bayes import GaussianNB

import pandas as pd

import csv

#importing data
trainingdata = pd.read_csv("train.csv")
#Checking columns and rows
print(trainingdata.head(10))
print(trainingdata.tail(3))
print(trainingdata.describe()) #allows us to identify missing values

#Processing and cleaning data: filling in missing values and converting categorical data to dummy variables

#Age

trainingdata["Age"] = trainingdata["Age"].fillna(trainingdata["Age"].median())


#Fare

trainingdata["Fare"] = trainingdata["Fare"].fillna(trainingdata["Fare"].median())


#Embarked

trainingdata["Embarked"] = trainingdata["Embarked"].fillna("S") #S is the most common embarked class
embarked_dummies = pd.get_dummies(trainingdata["Embarked"], prefix = "Embarked")
trainingdata = pd.concat([trainingdata, embarked_dummies], axis = 1)
trainingdata.drop("Embarked", axis = 1, inplace = True)

#Cabin

#trainingdata["Cabin"] = trainingdata["Cabin"].fillna("U") #U for unknown cabin - will likely cause issues if we try to fill cabin with the most common
#dummy encoding
#trainingdata["Cabin"] = trainingdata["Cabin"].map(lambda c : c[0] ) #remmapping the cabin name to the first letter of the cabin name i.e. index c[0]
#cabin_dummies = pd.get_dummies(trainingdata["Cabin"], prefix = "Cabin")
#appending cabin_dummies to trainingdata
#trainingdata = pd.concat([trainingdata, cabin_dummies], axis = 1)
#trainingdata.drop("Cabin", axis = 1, inplace = True)

#Sex

trainingdata["Sex"] = trainingdata["Sex"].map({"male":1, "female":0})

#Class

pclass_dummies = pd.get_dummies(trainingdata['Pclass'], prefix = "Pclass")
trainingdata = pd.concat([trainingdata, pclass_dummies], axis = 1)
trainingdata.drop(["PassengerId", "Name", "Ticket", "Pclass", "Cabin"], axis = 1, inplace = True) #Drop insignificant predictors


#loading test file
testdata = pd.read_csv("test.csv")

testdata["Age"] = testdata["Age"].fillna(testdata["Age"].median())


#Fare

testdata["Fare"] = testdata["Fare"].fillna(testdata["Fare"].median())


#Embarked

testdata["Embarked"] = testdata["Embarked"].fillna("S") #S is the most common embarked class
embarked_dummies = pd.get_dummies(testdata["Embarked"], prefix = "Embarked")
testdata = pd.concat([testdata, embarked_dummies], axis = 1)
testdata.drop("Embarked", axis = 1, inplace = True)

#Cabin

#testdata["Cabin"] = testdata["Cabin"].fillna("U") #fill in with the most frequent
#dummy encoding
#testdata["Cabin"] = testdata["Cabin"].map(lambda c : c[0] ) #remmapping the cabin name to the first letter of the cabin name i.e. index c[0]
#cabin_dummies = pd.get_dummies(testdata["Cabin"], prefix = "Cabin")
#appending cabin_dummies to testdata
#testdata = pd.concat([testdata, cabin_dummies], axis = 1)
#testdata.drop("Cabin", axis = 1, inplace = True)

#Sex

testdata["Sex"] = testdata["Sex"].map({"male":1, "female":0})

#Class

pclass_dummies = pd.get_dummies(testdata['Pclass'], prefix = "Pclass")
testdata = pd.concat([testdata, pclass_dummies], axis = 1)
PassID = testdata["PassengerId"]
testdata.drop(["PassengerId", "Name", "Ticket", "Pclass", "Cabin"], axis = 1, inplace = True)
print(testdata.head(1))

print(trainingdata.head(1))
train_cols = trainingdata.iloc[:,1:19] #setting survived
print(train_cols.head(20))

#Intialise LogisticRegression model
model = GaussianNB()

#Train the model
model.fit(train_cols, trainingdata["Survived"])
print("This is the training data!")
print(train_cols.head(1))
print("This below is the test data!")
print(testdata.head(1))
predicted = model.predict(testdata)
#output results
output = pd.DataFrame(columns = ["PassengerId", "Survived"])
output ["PassengerId"] = PassID
output ["Survived"] = predicted.astype(int)
output.to_csv("gausssian.csv", index = False)
