#Sheena Ganju, CS 4641 HW 1
#Decision Trees

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
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#k-means specific imports 
from sklearn import mixture
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

t0= time.clock()

# Fit a Gaussian mixture with EM using five components
gmm = mixture.GaussianMixture(n_components=10, covariance_type='full').fit(traindata)
print(gmm.means_)
print(gmm.lower_bound_)

##plot_results(traindata, gmm.predict(traindata), gmm.means_, gmm.covariances_, 0,
##             'Gaussian Mixture'

cv = train_test_split(X, Y, test_size=.30, random_state= 20)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.30, random_state= 20)

##Y_prediction = gmm.predict(X_test)
##
##from sklearn.metrics import log_loss
##loss = log_loss(Y_test, Y_prediction)*100
##print("Loss: " + str(loss))


t1= time.clock()
time = str(t1-t0)
print("Computation Time" + time)

##skplt.estimators.plot_learning_curve(gmm, X, Y, title = "Learning Curve: GMM")
##plt.show()




