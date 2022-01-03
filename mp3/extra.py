import os, h5py
from PIL import Image
import numpy as  np

def classify(Xtrain, Ytrain, Xdev, Ydev, Xtest):
    '''
    Ytest = classify(Xtrain, Ytrain, Xdev, Ydev, Xtest)

    Use any technique you like to train a classifier with the training set,
    and then return the correct class labels for the test set.
    Extra credit points are provided for beating various thresholds above 50%.

    Xtrain (NTRAIN x NDIM) - data matrix.  You may assume that NDIM > NTOKS
    Ytrain (NTRAIN) - list of class indices
    Xdev (NDEV x NDIM) - data matrix.
    Ydev (NDEV) - list of class indices
    Xtest (NTEST x NDIM) - data matrix.
    '''
    mu = todo_dataset_mean(Xtrain)
    ctrain, cdev, ctest= todo_center_datasets(Xtrain,Xdev,Xtest,mu)
    V, Lambda = todo_find_transform(Xtrain)
    ttrain, tdev, ttest = todo_transform_datasets(ctrain, cdev, ctest, V)
    bestsize, accuracies = todo_find_bestsize(ttrain, tdev, ttest, Ydev, Lambda)
    D = todo_distances(ttrain,ttest,bestsize)
    hyps = todo_nearest_neighbor(Ytrain,D)

    return hyps

def todo_dataset_mean(X):
    '''
    mu = todo_dataset_mean(X)
    Compute the average of the rows in X (you may use any numpy function)
    X (NTOKSxNDIMS) = data matrix
    mu (NDIMS) = mean vector
    '''
    return np.average(X, axis=0)


def todo_center_datasets(train, dev, test, mu):
    '''
    ctrain, cdev, ctest = todo_center_datasets(train, dev, test, mu)
    Subtract mu from each row of each matrix, return the resulting three matrices.
    '''
    ctrain = train.copy()
    cdev = dev.copy()
    ctest = test.copy()
    for i in range(ctrain.shape[0]):
        ctrain[i,:] -= mu
    for i in range(cdev.shape[0]):
        cdev[i,:] -= mu
    for i in range(cdev.shape[0]):
        cdev[i,:] -= mu
    return ctrain, cdev, ctest
    

def todo_find_transform(X):
    '''
    V, Lambda = todo_find_transform(X)
    X (NTOKS x NDIM) - data matrix.  You may assume that NDIM > NTOKS
    V (NDIM x NTOKS) - The first NTOKS principal component vectors of X
    Lambda (NTOKS) - The  first NTOKS eigenvalues of the covariance or gram matrix of X

    Find and return the PCA transform for the given X matrix:
    a matrix in which each column is a principal component direction.
    You can assume that the # data is less than the # dimensions per vector,
    so you should probably use the gram-matrix method, not the covariance method.
    Standardization: Make sure that each of your returned vectors has unit norm,
    and that its first element is non-negative.
    Return: (V, Lambda)
      V[:,i] = the i'th principal component direction
      Lambda[i] = the variance explained by the i'th principal component

    V and Lambda should both be sorted so that Lambda is in descending order of absolute
    value.  Notice: np.linalg.eig doesn't always do this, and in fact, the order it generates
    is different on my laptop vs. the grader, leading to spurious errors.  Consider using 
    np.argsort and np.take_along_axis to solve this problem, or else use np.linalg.svd instead.
    '''
    gram = np.dot(X,X.T)
    w, U = np.linalg.eig(gram)
    V_ = np.dot(X.T,U)
    V = np.zeros(V_.shape)
    temp = np.abs(w)
    order = np.flip(np.argsort(temp))
    Lambda = np.take_along_axis(w, order, axis=0)
    '''

    for x in range(len(V_)):
        for y in range(len(V_[0])):
            #max_ = V_[x].max()
            #min_ = V_[x].min()
            V_[x][y] = (V_[x][y]-min_) / (max_ - min_)
            V[x][y] = V_[x][order[y]]
            '''
    for i in range(len(V_[0])):
        idx = order[i]
        if V_[0,i] < 0:
            V[:,idx] = V_[:,i]/np.linalg.norm(V_[:,i]) * -1
        else:
            V[:,idx] = V_[:,i]/np.linalg.norm(V_[:,i])
    
    return V, Lambda
    
    

def todo_transform_datasets(ctrain, cdev, ctest, V):
    '''
    ttrain, tdev, ttest = todo_transform_datasets(ctrain, cdev, ctest, V)
    ctrain, cdev, ctest are each (NTOKS x NDIMS) matrices (with different numbers of tokens)
    V is an (NDIM x K) matrix, containing the first K principal component vectors
    
    Transform each x using transform, return the resulting three datasets.
    '''
    return np.dot(ctrain,V), np.dot(cdev,V), np.dot(ctest,V)

def todo_distances(train,test,size):
    '''
    D = todo_distances(train, test, size)
    train (NTRAINxNDIM) - one training vector per row
    test (NTESTxNDIM) - one test vector per row
    size (scalar) - number of dimensions to be used in calculating distance
    D (NTRAIN x NTEST) - pairwise Euclidean distances between vectors

    Return a matrix D such that D[i,j]=distance(train[i,:size],test[j,:size])
    '''
    D = np.zeros((train.shape[0],test.shape[0]))
    for i in range(train.shape[0]):
        for j in range(test.shape[0]):
            distance = train[i,:size] - test[j,:size]
            D[i,j] = np.sqrt(np.dot(distance, distance.T))
    return(D)

def todo_nearest_neighbor(Ytrain, D):
    '''
    hyps = todo_nearest_neighbor(Ytrain, D)
    Ytrain (NTRAIN) - a vector listing the class indices of each token in the training set
    D (NTRAIN x NTEST) - a matrix of distances from train to test vectors
    hyps (NTEST) - a vector containing a predicted class label for each test token

    Given the dataset train, and the (NTRAINxNTEST) matrix D, returns
    an int numpy array of length NTEST, specifying the person number (y) of the training token
    that is closest to each of the NTEST test tokens.
    '''
    ntrian,ntest = D.shape
    hyps = np.zeros(ntest)
    near = 0
    for i in range(ntest):
        near = D[0][i]
        for j in range(ntrian):
            if(D[j,i] <= near):
                near = D[j,i]
                hyps[i] = Ytrain[j]
    return hyps

def todo_compute_accuracy(Ytest, hyps):
    '''
    ACCURACY, CONFUSION = todo_compute_accuracy(TEST, HYPS)
    TEST (NTEST) - true label indices of each test token
    HYPS (NTEST) - hypothesis label indices of each test token
    ACCURACY (scalar) - the total fraction of hyps that are correct.
    CONFUSION (4x4) - confusion[ref,hyp] is the number of class "ref" tokens (mis)labeled as "hyp"
    '''
    confusion = np.zeros((4,4))
    accuracy = 0
    acc = 0
    for i in range(len(hyps)):
        hyp = hyps[i]
        ref = Ytest[i]
        confusion[int(ref),int(hyp)] += 1
    for i in range(4):
        acc += confusion[i,i]
    accuracy = acc / len(hyps)
    return accuracy, confusion

def todo_find_bestsize(ttrain, tdev, Ytrain, Ydev, variances):
    '''
    BESTSIZE, ACCURACIES = todo_find_bestsize(TTRAIN, TDEV, YTRAIN, YDEV, VARIANCES)
    TTRAIN (NTRAINxNDIMS) - training data, one vector per row, PCA-transformed
    TDEV (NDEVxNDIMS)  - devtest data, one vector per row, PCA-transformed
    YTRAIN (NTRAIN) - true labels of each training vector
    YDEV (NDEV) - true labels of each devtest token
    VARIANCES - nonzero eigenvectors of the covariance matrix = eigenvectors of the gram matrix

    BESTSIZE (scalar) - the best size to use for the nearest-neighbor classifier
    ACCURACIES (NTRAIN) - accuracy of dev classification, as function of the size of the NN classifier

    The only sizes you need to test (the only nonzero entries in the ACCURACIES
    vector) are the ones where the PCA features explain between 92.5% and
    97.5% of the variance of the training set, as specified by the provided
    per-feature variances.  All others should be zero.
    '''
    accuracies = np.zeros(len(Ytrain))
    var_s = np.sum(variances)
    sum = 0
    for i in range(len(Ytrain)):
        sum += variances[i]
        accuracies[i] = 0
        if ((sum/var_s > 0.925) and (sum/var_s < 0.975)):
            distance = todo_distances(ttrain,tdev,i)
            hyps = todo_nearest_neighbor(Ytrain, distance)
            accuracy, confusion = todo_compute_accuracy(Ydev, hyps)
            accuracies[i] = accuracy
    bestsize = np.argmax(accuracies)
    return int(bestsize), accuracies
