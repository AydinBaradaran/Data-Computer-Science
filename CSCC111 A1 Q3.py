import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics
from sklearn.metrics import f1_score

def LogReg(X_train, X_test, y_train, y_test):
    w = np.zeros((2,1));
    b = np.zeros((1,1));
    X_train = X_train[0,1];
    X_test = X_test[0,1];
    z = (np.matmul(X_train,w.reshape(-1,1))+b);
    A = 1/(1+np.exp(-z));
    loss = -np.mean(y_train*np.log(A) + (1-y_train)*np.log(1-A));
    plt.plot(A,loss);
    return 0;
    


def GradientDescent(X_train, X_test, y_train, y_test,alpha,T):
    w = np.zeros((4,1));
    b = np.zeros((1,1));
    # matrix for gradient descent
    matrix = [];
    # matrix for sigmoid function
    matrix2 = [];
    # matrix for the loss function
    matrix3 = [];
    iterations = 0;
    for i in range(80):
        # z equation in the notes
        z = (np.matmul(X_train,w.reshape(-1,1))+b);
        # sig function
        sig = 1/(1+np.exp(-z));
        # loss function in the notes
        loss = -np.mean(y_train*np.log(sig) + (1-y_train)*np.log(1-sig));
        # derivative of variable z
        dz = sig - y_Train;
        # derivative of variabel w
        dw = 1/20 * (np.matmul(X_train.T, dz));
        # gradient descent equation
        w = w - (alpha)*dw;
        matrix.append(w);
        matrix2.append(sig);
        matrix3.append(loss);
        #print(matrix);
        # checks the number of iterations
        if i > 0:
            # check the differences aren't less than the threshold, if it is save the iteration count
            if (((sum(matrix[i-1])/4) - (sum(matrix[i]))/4)/4) < T:
                iteration = i;
                
    # fscore done here after the gradient descent
    prediction_matrix = [];
    for i in matrix2:
        if i[0] > 0.5:
            prediction_matrix.append(1);
        else:
            prediction_matrix.append(0);
    
    f_score = f1_score(prediction_matrix,y_Train);    
    return iterations,f_score;


if __name__ == "__main__":
    iris = datasets.load_iris();
    # take the first two classes of the dataset i.e., first 100 instances.
    X = iris.data[:100,:]
    y = iris.target[:100] # the labels
    X_train, X_test, y_Train, y_Test = train_test_split(X, y,test_size=0.2, train_size= 0.8);
    y_Train = y_Train.reshape(-1,1);
    y_Test = y_Test.reshape(-1,1);
    #LogReg(X_train, X_test, y_Train, y_Test);
    iteration1, f_score1 = GradientDescent(X_train, X_test, y_Train, y_Test,0.01,0.0001);
    print("alpha = 0.01 with T = 0.0001 has iterations and fscore respectively as ",iteration1," and ",f_score1);
    iteration2, f_score2 = GradientDescent(X_train, X_test, y_Train, y_Test,0.01,0.9);
    print("alpha = 0.01 with T = 0.9 has iterations and fscore respectively as ",iteration2," and ",f_score2);
    iteration3, f_score3 = GradientDescent(X_train, X_test, y_Train, y_Test,0.5,0.0001);
    print("alpha = 0.5 with T = 0.0001 has iterations and fscore respectively as ",iteration3," and ",f_score3);
    iteration4, f_score4 = GradientDescent(X_train, X_test, y_Train, y_Test,0.5,0.9);
    print("alpha = 0.5 with T = 0.9 has iterations and fscore respectively as ",iteration4," and ",f_score4);
    iteration5, f_score5 = GradientDescent(X_train, X_test, y_Train, y_Test,0.99,0.0001);
    print("alpha = 0.99 with T = 0.0001 has iterations and fscore respectively as ",iteration5," and ",f_score5);
    iteration6, f_score6 = GradientDescent(X_train, X_test, y_Train, y_Test,0.99,0.9);
    print("alpha = 0.99 with T = 0.9 has iterations and fscore respectively as ",iteration6," and ",f_score6);