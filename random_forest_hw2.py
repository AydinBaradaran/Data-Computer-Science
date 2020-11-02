import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import genfromtxt
from mpl_toolkits import mplot3d
from numpy import linalg as LA
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor

from collections import Counter



def rand_forest_maker(trainsTestSet,trainsTrainSet,testsTestSet,testsTrainSet):
    randForest = RandomForestClassifier(n_estimators=1000);
    randForest.fit(trainsTrainSet, trainsTestSet);
    predict = randForest.predict(testsTrainSet)
    return predict;



if __name__ == "__main__":
    traincsv = pd.read_csv("train.csv");
    testcsv = pd.read_csv("test.csv",skiprows=1);
    #f = open("customer.csv");
    # remove name, convert male and female to ints as well as emabarked to 1,2,3
    trainData = pd.DataFrame(traincsv);
    trainData = trainData.dropna();
    trainData = trainData.drop(columns="Name");
    trainData.Sex[trainData.Sex=='male'] = 1;
    trainData.Sex[trainData.Sex=='female'] = 0;
    trainData.Embarked[trainData.Embarked =='S'] = 0;
    trainData.Embarked[trainData.Embarked =='Q'] = 1;
    trainData.Embarked[trainData.Embarked =='C'] = 2;    
    # drop unnecesarry strings
    trainData = trainData.drop(columns="Ticket");    
    trainData = trainData.drop(columns="Cabin");
    # same as the above trtain file done here
    testData = pd.DataFrame(testcsv);
    testData = testData.dropna();
    testData = testData.drop(columns="Name");
    testData.Sex[testData.Sex=='male'] = 1;
    testData.Sex[testData.Sex=='female'] = 0;
    testData.Embarked[testData.Embarked =='S'] = 0;
    testData.Embarked[testData.Embarked =='Q'] = 1;
    testData.Embarked[testData.Embarked =='C'] = 2;     

    #unnecssary string columns droped
    testData = testData.drop(columns="Ticket");
    testData = testData.drop(columns="Cabin");    
    
    
    # convert to ndarray
    trainsTestSet = (trainData.Survived).values;
    trainsTrainSet = (trainData.drop(columns='Survived')).values;
    testsTestSet = (testData.Survived).values
    testsTrainSet = (testData.drop(columns='Survived')).values;
    
    # create a random forest
    randomForest = rand_forest_maker(trainsTestSet,trainsTrainSet,testsTestSet,testsTrainSet);
    # get the confusiuon matrix for it 
    conf = confusion_matrix(testsTestSet,randomForest);
    print(conf);