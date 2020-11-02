import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.interpolate import Rbf
from sklearn.datasets import load_boston
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn import utils
from sklearn.metrics import classification_report, confusion_matrix
from numpy import linalg as LA
from sklearn.linear_model import Ridge
#from rbf.interpolate import RBFInterpolant


bostondataset = load_boston()  
boston = pd.DataFrame(bostondataset.data, columns=bostondataset.feature_names)
X = boston[['RM', 'AGE', 'DIS', 'RAD', 'TAX']]  # take 5 features
y = bostondataset.target  # target value
y = y.astype('int');
X_sub = np.asarray(X['RM']);
X_sub2 = np.asarray(np.transpose(np.transpose(X)[0:2]));

rbf = SVC(kernel = 'rbf', degree = 3);
predict = rbf.fit(X_sub.reshape(-1,1),y).predict(X_sub.reshape(-1,1));
predict2 = rbf.fit(X_sub2,y).predict(X_sub2);
error = LA.norm(y-predict2);
error = error**2;
print("without ridge regression error is ",error);
ridge_reg = Ridge();

      