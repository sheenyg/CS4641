#Sheena Ganju, CS 4641 HW 3

#info from http://scikit-learn.org/stable/modules/
#generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans

#import sklearn statements
import sklearn as sklearn
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split

#for graph from http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import random_projection
from sklearn.metrics import accuracy_score

#other imports 
import scikitplot as skplt
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import validation_curve
from datetime import date

#Read data in using pandas
trainDataSet = pd.read_csv("geoplaces2.csv", sep = ',', header = None, low_memory = False)

#encode text data to integers using getDummies
traindata = pd.get_dummies(trainDataSet)
# Create decision Tree using major_category, month, year, to predict violent or not 
# train split uses default gini node, split using train_test_split

X = traindata.values[1:, 1:]
Y = traindata.values[1:,0]

#Finding the optimal component #

complist = [2,3,4,5,6]

for each in complist:
    comp = each
    t0= time.clock()

    print("Time started")
    # Fit the PCA analysis
    result = random_projection.GaussianRandomProjection(n_components=comp).fit(traindata)

    print("Components"+ str(result.components_))
    print("Explained Variance"+ str(result.explained_variance_))
    print("Explained Variance Ration" + str(result.explained_variance_ratio_))
    print("RP Score"+ str(result.score(traindata, y= None)))

    t1= time.clock()
    timetaken = str(t1-t0)
    print("Computation Time" + timetaken)


###Graphical Representation of n = 3
### Fit the PCA analysis
##result = PCA(n_components=3).fit(traindata)
##from matplotlib.mlab import PCA as mlabPCA
##
##mlab_pca = mlabPCA(traindata.T)
##
##print('PC axes in terms of the measurement axes scaled by the standard deviations:\n', mlab_pca.Wt)
##
##plt.plot(mlab_pca.Y[0:20,0],mlab_pca.Y[0:20,1], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
##plt.plot(mlab_pca.Y[20:40,0], mlab_pca.Y[20:40,1], '^', markersize=7, color='red', alpha=0.5, label='class2')
##
##plt.xlabel('x_values')
##plt.ylabel('y_values')
##plt.xlim([-4,4])
##plt.ylim([-4,4])
##plt.legend()
##plt.title('Transformed samples with class labels from matplotlib.mlab.PCA()')
##
##plt.show()

