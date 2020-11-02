import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from numpy import linalg as LA
from pandas import Series

#csvdata = pd.read_csv("hw1_polyreg.csv");
#f = open("hw1_polyreg.csv");
#data1 = pd.DataFrame(csvdata);

#X_train, X_test, y_train, y_test = train_test_split(data1["x"],data1["y"], test_size=0.25, train_size= 0.75);

def FitPolynomialRegression(K,x,y):
    # x matrix
    matrix = np.ndarray(shape=(len(x),K+1),dtype = float);
    for i in range(0,len(x)):
        for j in range (0,K+1):
            matrix[i,j] = (x.index[i])**j;
    # inverse operations
    matrixTmatrix = np.matmul(np.transpose(matrix),matrix);
    matrix_inverse = np.linalg.inv(matrixTmatrix);
    matrixTy = np.matmul(np.transpose(matrix),y);
    # weight matrix being returned
    w = (np.matmul(matrix_inverse, matrixTy));
    return w;

def EvalPolynomial(x,w):
    # prediction at each input, this is a vector for it.
    y = np.ndarray(shape=(len(w),len(x)), dtype = float);
    for i in range (0,len(w)):
        for j in range(0,len(x)):
            y[i][j] = (((x.index.values[j])**(i)));
    y = np.matmul(w,y);
    y = y.reshape(1,len(x));
    return y;

def GetBestPolynomial(xTrain, yTrain, xTest, yTest, h):
    # two variables will be the mse for the test and train sets
    MSE_test_degree = np.ndarray(shape=h, dtype = float);
    MSE_train_degree = np.ndarray(shape=h, dtype = float);
    # loop through each power indivdually as per the handout
    for i in range(1,h+1):
        # fit the training data
        train_weights = FitPolynomialRegression(i,xTrain, yTrain);
        # fit the test data
        test_weights = FitPolynomialRegression(i,xTest, yTest);
        # evaluate the train
        y_train_eval = EvalPolynomial(xTrain,train_weights);
        # evaluate the test
        y_test_eval = EvalPolynomial(xTest,test_weights);
        # calculate the mse for power i for train error
        MSE_train_degree[i-1] = ((LA.norm(yTrain-y_train_eval.reshape(75,)))**2)/len(yTrain);
        # calculate the mse for power i for test error
        MSE_test_degree[i-1] = ((LA.norm(yTest-y_test_eval.reshape(25,)))**2)/len(yTest);
    # below is the plot followed by the legend
    degree_axis = range(1,h+1);
    plt.plot(degree_axis, MSE_test_degree, label = "Test MSE");
    plt.plot(degree_axis, MSE_train_degree, label = "Train MSE");
    plt.legend(loc = "upper right");
    return MSE_test_degree,MSE_train_degree;

if __name__ == "__main__":
    csvdata = pd.read_csv("hw1_polyreg.csv");
    f = open("hw1_polyreg.csv");
    data1 = pd.DataFrame(csvdata);
    X_train, X_test, y_train, y_test = train_test_split(data1["x"],data1["y"], test_size=0.25, train_size= 0.75);    
    mse_test, mse_train = GetBestPolynomial(X_train,y_train,X_test,y_test,10);
    for i in range(0,len(mse_test)):
        print("for degree",i+1," the mean square error for test error is ", mse_test[i]);
    print("---------------------------------------------------------------------------------")
    for i in range(0,len(mse_train)):
        print("for degree",i+1," the mean square error for train error is ", mse_train[i]);    
    
    
