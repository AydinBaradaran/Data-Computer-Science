import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import genfromtxt
from mpl_toolkits import mplot3d
from numpy import linalg as LA
import random



def my_kmeans(features,k,kmeansplus):
    if kmeansplus==True:
        cluster = kmeanspluspplus(features,k);
    else:
        cluster = my_kmeans_normal(features,k);
    return cluster;

def my_kmeans_normal(features,k):
    #this array helps check for convergence
    intialCluster=np.ndarray(shape=(k,3),dtype = 'float64');
    # this is the final product cluster
    cluster = np.ndarray(shape=(k,3),dtype = 'float64');
    # this counter helsp to make sure the intial center is set first as the comparison
    counter = 0;
    # condition for the while loop
    check = True;
    # this loop ensures we're able to get convergence
    while check is True:
        # always set the intial which is the previous center to be compare to the new cluster in this iteration
        intialCluster = cluster;
        # this is the formula that saves the distance between each point and its centre
        normdist = np.ndarray(shape=(200,k),dtype = 'float64');
        # this is the random centre in the range of each feature
        centroids = np.ndarray(shape=(k,3),dtype = 'float64');
        # this is r_i in the kmeans algo
        upcent = np.ndarray(shape=(200,1),dtype = 'float64');
        # array used to store the random values for the range of each feature
        featureVector = np.ndarray(shape=(k,3),dtype = 'float64');
        # saves the top as a whole, but as the right shape for the matrix
        referenceVector = np.zeros(shape=(k,3),dtype = 'float64');
        # the three lines is doing the random values in range for the intial centres
        featureOne = np.random.randint(np.amin(features[:,1]), np.amax(features[:,1]), size=k)
        featureTwo = np.random.randint(np.amin(features[:,2]), np.amax(features[:,2]), size=k) 
        featureThree = np.random.randint(np.amin(features[:,3]), np.amax(features[:,3]), size=k)
        # save each one as a column
        referenceVector[:,0] = featureOne;
        referenceVector[:,1] = featureTwo;
        referenceVector[:,2] = featureThree;
        # save it to a centroid
        centroids = referenceVector;
        # loop through each entry to populate the normdist matrix
        for j in range(0,k):
            for i in range(0,200):
                # the argmin formula in the notes
                normdist[i][j] = np.linalg.norm(centroids[j] - features[:,1:4][i])**2
        # here we figure out the shortest distance and assign it to what cluster it belongs to
        for i in range(0,200):
            upcent[i] = np.argmin(normdist[i]);
        # count the number of each centre being the min
        countfeature = np.unique(upcent, return_counts=True);
        sumVector = np.zeros(shape=(k,3),dtype = 'float64');
        # loop through and sum the values that correspond for each cluster
        for j in range(0,k):
            for i in range (1,200):
                if upcent[i] == j:
                    sumVector[j] = sumVector[j] + features[:,1:4][i]
                else:
                    None
        #print(sumVector)
        for i in range(0,k):
            # here we average each cluster
            cluster[i] = sumVector[i]/countfeature[1][i]
        # check if this is the first run
        if (counter==0):
            counter+=1;
        # else we just loop through to find one instance where the values dont equal
        # this will stop the loop
        else:
            check = False;
            for i in range(0,k):
                for j in range(0,cluster.shape[1]):
                    if(intialCluster[i][j] != cluster[i][j]):
                        check = True;
    return cluster;

def my_kmeans_plot(clusters):
    ax = plt.axes(projection="3d")
    ax.plot3D(clusters["Age"],clusters["Gender"],clusters["Annual Income (k$)"])
    for i in range(0,clusters.shape[0]):
        ax.plot(clusters[i]);
    return None;
    
    
def kmeanspluspplus(features,k):
    
    centres = np.ndarray(shape=(k,3),dtype = 'float64');
    # the return cluster
    cluster = np.ndarray(shape=(k,3),dtype = 'float64');
    # the random point we pick as our centre
    intialCentre = random.choice(features[:,1:4])
    # distance i.e D(y_i)
    distance = np.ndarray(shape=(200,1),dtype = 'float64');
    # counter for the while loop so we get k clusters
    kCounter = 0;
    # set the random point as our first cluster 
    cluster[kCounter] = intialCentre;
    # add one so we dont replace it
    kCounter = kCounter + 1;
    # loop through to make k clusters
    while kCounter < k:
        # reset distance for each run
        distance = np.zeros(shape=(200,1),dtype = 'float64');
        # loop through to make D(y_i)
        for j in range(0,200):
            distance[j] =  np.linalg.norm(cluster[kCounter-1] - features[:,1:4][j])**2
        # create trhe prob distn
        prob = np.zeros(shape=(200,),dtype = 'float64');
        # the formula is step 3 of the kmeans++ formula 
        for i in range(0,200):
            prob[i] = (distance[i]**2)/sum(distance**2)
        # add in the cluster to be returned 
        cluster[kCounter] = (features[:,1:4])[np.random.choice(200, 1, p = prob)]
        # update the centre for step 2 of the next iteation
        intialCentre = cluster[kCounter];
        # update counter
        kCounter = kCounter + 1;
    return cluster;

def kmeans():
    return None

if __name__ == "__main__":
    csvdata = pd.read_csv("customer.csv",skiprows=1);
    f = open("customer.csv");
    data1 = pd.DataFrame(csvdata);
    data1.Gender[data1.Gender =='Male'] = 1
    data1.Gender[data1.Gender =='Female'] = 0
    features = data1.values
    ax = plt.axes(projection="3d")
    ax.plot3D(features["Age"],features["Gender"],features["Annual Income (k$)"])
    #A
    cluster1 = my_kmeans(features,2,False)
    cluster2 = my_kmeans(features,3,False)
    cluster3 = my_kmeans(features,4,False)
    cluster4 = my_kmeans(features,5,False)
    #B i.e kmeans++
    cluster5 = my_kmeans(features,2,True)
    cluster6 = my_kmeans(features,3,True)
    cluster7 = my_kmeans(features,4,True)
    cluster8 = my_kmeans(features,5,True)    
    
    
    
    
    
    
        