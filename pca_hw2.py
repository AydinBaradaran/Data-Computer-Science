import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import genfromtxt
from mpl_toolkits import mplot3d
from numpy import linalg as LA
from sklearn import datasets



def my_pca(data_matrix,k):
    # covariance matrix
    covMatrix = np.cov(data_matrix.T);
    # eigen value
    eigenValues = np.ndarray(shape=(1,4),dtype = 'float64');
    #eigen vector
    eigenVectors = np.ndarray(shape=(4,4),dtype = 'float64');
    eigenVectors = LA.eig(covMatrix);
    # sorted eiogne values
    eigenValues = np.asanyarray(LA.eigvals(covMatrix));
    eigenValues.sort();
    eigenValues = eigenValues[::-1]
    #sorted eigen vectors
    eigenVectors = eigenVectors[1];
    eigenVectors.sort();
    eigenVectors = eigenVectors[::-1]
    #max eigne vectors
    topEigen = np.ndarray(shape=(4,k),dtype = 'float64');
    low_dim_matrix = np.ndarray(shape=(150,k),dtype = 'float64');
    # loop through taking the k highets eigne vectors
    for i in range(0,k):
        topEigen[:,i] = eigenVectors[i];
    # equation of data matrix times eigen vectors(highest ones)
    low_dim_matrix = np.matmul(data_matrix,topEigen);
    return low_dim_matrix;

def my_pca_plot(low_dim_matrix):
    if low_dim_matrix.shape == (150, 2):
        plt.scatter(low_dim_matrix[:,0],low_dim_matrix[:,1])
    else:
        ax = plt.axes(projection="3d")
        ax.plot3D(low_dim_matrix[:,0],low_dim_matrix[:,1],low_dim_matrix[:,2])
    return None;


if __name__ == "__main__":
    iris = datasets.load_iris();
    X = iris.data;
    low_dim_matrix = my_pca(X,2);
    low_dim_matrix2 = my_pca(X,3);
    
    print(low_dim_matrix)
    print(low_dim_matrix2)
    print(low_dim_matrix.shape)
    my_pca_plot(low_dim_matrix)
    my_pca_plot(low_dim_matrix2)
    
      