#Import relevant libraries - may get deprecated warnings - these libraries are only used to tune hyperparatmeters and code will still run regardless
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score

import csv

#importing data
trainingdata = pd.read_csv("train.csv")

#Processing and cleaning data: filling in missing values and converting categorical data to dummy variables

#Age
trainingdata["Age"] = trainingdata["Age"].fillna(trainingdata["Age"].median())


#Fare
trainingdata["Fare"] = trainingdata["Fare"].fillna(trainingdata["Fare"].median())


#Embarked
trainingdata["Embarked"] = trainingdata["Embarked"].fillna("S")                    #Fill missing value with most common
embarked_dummies = pd.get_dummies(trainingdata["Embarked"], prefix = "Embarked")   #Convert categories to dummy variables
trainingdata = pd.concat([trainingdata, embarked_dummies], axis = 1)               #Add dummy variable columns to trainingdata dataframe
trainingdata.drop("Embarked", axis = 1, inplace = True)                            #Remove original embarked columns


#Sex
trainingdata["Sex"] = trainingdata["Sex"].map({"male":1, "female":0})              #Map male to 1 and female to 0

#Extract Titles from the Name Column, Map each name to a category, get dummy variables for each category
trainingdata["Title"] = trainingdata["Name"].map(lambda name:name.split(",")[1].split(".")[0].strip()) #Extract the title
Title_Dictionary = {"Capt":       "Officer",                                                           #Dictionary containing all possible names, mapped to a category
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"

                    }
trainingdata["Title"] = trainingdata.Title.map(Title_Dictionary)                                       #Replace titles with categories in Dictionary
title_dummies = pd.get_dummies(trainingdata["Title"], prefix = "Title")                                #Convert titles to dummy variables
trainingdata = pd.concat([trainingdata, title_dummies], axis = 1)                                      #Add title dummy variables to the trainingdata DataFrame
trainingdata.drop(["Title"], axis = 1, inplace = True)                                                 #Remove original Title column

#Class
pclass_dummies = pd.get_dummies(trainingdata['Pclass'], prefix = "Pclass")                             #Convert passenger classes to dummy variables
trainingdata = pd.concat([trainingdata, pclass_dummies], axis = 1)                                     #Add passenger dummy variable columns to trainingdata DataFrame
trainingdata.drop(["Pclass"], axis = 1, inplace = True)                                                #Remove original Pclass Column

trainingdata.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis = 1, inplace = True)                #Remove insignificant predictors

#Load test file
testdata = pd.read_csv("test.csv")

#Applying same operations as above.
#Age
testdata["Age"] = testdata["Age"].fillna(testdata["Age"].median())

#Fare
testdata["Fare"] = testdata["Fare"].fillna(testdata["Fare"].median())


#Embarked
testdata["Embarked"] = testdata["Embarked"].fillna("S") #S is the most common embarked class
embarked_dummies = pd.get_dummies(testdata["Embarked"], prefix = "Embarked")
testdata = pd.concat([testdata, embarked_dummies], axis = 1)
testdata.drop("Embarked", axis = 1, inplace = True)

#Sex
testdata["Sex"] = testdata["Sex"].map({"male":1, "female":0})

#Get titles
testdata['Title'] = testdata['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
Title_Dictionary = {"Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"}
testdata["Title"] = testdata.Title.map(Title_Dictionary)
title_dummies = pd.get_dummies(testdata["Title"], prefix = "Title")
testdata.drop(["Title"], axis = 1, inplace = True)
testdata = pd.concat([testdata, title_dummies], axis = 1)

#Class
pclass_dummies = pd.get_dummies(testdata['Pclass'], prefix = "Pclass")
testdata = pd.concat([testdata, pclass_dummies], axis = 1)
PassID = testdata["PassengerId"]                                                 #Setting PassID from test data for easy input into outgoing .csv file later
testdata.drop(["PassengerId", "Name", "Ticket", "Pclass", "Cabin"], axis = 1, inplace = True)

#Setting predictors - target column set later when intialising model
train_cols = trainingdata.iloc[:,1:18]

#set to true to run parameter search - This loop will go through all the possibe paraters set in paramter_grid and find the best set based on highest accuracy score
run_gs = False

if run_gs:
    parameter_grid = {
                 'max_depth' : [4, 6, 8],
                 'n_estimators': [50, 10],
                 'max_features': ['sqrt', 'auto', 'log2'],
                 'min_samples_split': [2, 3, 10],
                 'min_samples_leaf': [1, 3, 10],
                 'bootstrap': [True, False],
                 }
    forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(trainingdata["Survived"], n_folds=5)

    grid_search = GridSearchCV(forest,
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=cross_validation)

    grid_search.fit(train_cols, trainingdata["Survived"])
    model = grid_search
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
else:
    parameters = {'bootstrap': True, 'min_samples_leaf': 1, 'n_estimators': 10,                 #Using parameters found using search above
                  'min_samples_split': 10, 'max_features': 'auto', 'max_depth': 6}

    model = RandomForestClassifier(**parameters)
    model.fit(train_cols, trainingdata["Survived"])                                             #Fitting model using model.fit(X,y), where X are the predictors, and y the target

    predicted = model.predict(testdata)
    #Output results to csv
    output = pd.DataFrame(columns = ["PassengerId", "Survived"])
    output ["PassengerId"] = PassID
    output ["Survived"] = predicted.astype(int)
    output.to_csv("RandomForestClassifierSubmit.csv", index = False)
