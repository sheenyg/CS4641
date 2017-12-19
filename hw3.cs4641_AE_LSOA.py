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
from sklearn.model_selection import ShuffleSplit

#for graph from http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
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
from sknn import ae, mlp

#Read data in using pandas
trainDataSet = pd.read_csv("london_crime_by_lsoa.csv", sep = ',', header = None, low_memory = False)

#encode text data to integers using getDummies
traindata = pd.get_dummies(trainDataSet)
traindata= traindata[:1000]
traindata= traindata.values
# train split uses default gini node, split using train_test_split

X = traindata[1:, 1:]
Y = traindata[1:,0]
cv = train_test_split(X, Y, test_size=.33, random_state= 20)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.33, random_state= 20)


#Finding the optimal component 

AELayers = [ae.Layer("Sigmoid", units = 1000), ae.Layer("Sigmoid", units=500), ae.Layer("Sigmoid", units=250)]
NNLayers = [mlp.Layer("Sigmoid", units = 1000), mlp.Layer("Sigmoid", units = 500), mlp.Layer("Softmax", units= 15)]



##
##for each in complist:
##    comp = each
t0= time.clock()

print("Time started")
# Fit the Autoencoder

result = ae.AutoEncoder(AELayers,warning=None, random_state=0, learning_rule=u'sgd', learning_rate=0.1, learning_momentum=0.9,
                        regularize=None, weight_decay=None, dropout_rate=None, batch_size=1, n_iter=None,
                        n_stable=10, f_stable=0.001, valid_set=None, valid_size=0.0, loss_type=None, debug=False, verbose=None).fit(traindata)


t1= time.clock()
timetaken = str(t1-t0)
print("Computation Time" + timetaken)

#autoencoder results
result = result.transform(traindata)
x_test = X_test

#PLot data from blog.keras.io
# use Matplotlib (don't ask)

# Fit the PCA analysis
plt.plot(traindata, 'o', markersize=2, color='blue', alpha=0.5)
plt.plot(traindata[0], 'o', markersize=2, color='blue', alpha=0.5, label = "original")
plt.plot(result, '^', markersize=2, color='red', alpha=0.5)
plt.plot(result[0], '^', markersize=2, color='red', alpha=0.5, label = "autoencoder")
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.xlim([0,1000])
plt.ylim([-1.5,3])
plt.legend()
plt.title('Original vs. Transformed Data, AutoEncoders')
plt.show()
