#Sheena Ganju, CS 4641 HW 1
#Nueral network implementation using scikit learn,
#help from http://www.kdnuggets.com/2016/10/beginners-guide-neural-networks-python-scikit-learn.html/2

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import csv
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
import scikitplot as skplt
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from matplotlib.colors import ListedColormap

#read in data
#Read data in using pandas

trainDataSet = pd.read_csv("london_crime_by_lsoa.csv", sep=",", header= None, low_memory= False)
print("Dataset: ", trainDataSet.head())

#encode text data to integers using getDummies
traindata = pd.get_dummies(trainDataSet)

# Create decision Tree using major_category, month, year, to predict violent or not 
# train split uses default gini node, split using train_test_split

X = traindata.values[1:,:3]
Y = traindata.values[1:,4]

#start timer
t0= time.clock()


#find the best N by plotting
kArray = [1, 5, 8, 10, 20, 30]
ansArray = []
for each in kArray:
    clf = KNeighborsClassifier(n_neighbors = each)
    clf.fit(X_train, Y_train)
    f= clf.predict(X_test)
    g = accuracy_score(f, Y_test)*100
    ansArray.append(g)
    
fig = plt.figure()
ax= fig.add_subplot(111)
ax.set_title("K vs Accuracy")
ax.plot(kArray, ansArray)
plt.show()


clf = KNeighborsClassifier(n_neighbors = 5, weights = "distance")

clf.fit(X,Y)
print("Classifier score, training" + str(clf.score(X_train, Y_train)))
print("Classifier score, testing" + str(clf.score(X_test, Y_test)))

train_prediction = clf.predict(X_train)
trainaccuracy = accuracy_score(train_prediction, Y_train)*100
print("The training accuracy for this is " +str(trainaccuracy))

#output
Y_prediction = clf.predict(X_test)
accuracy = accuracy_score(Y_test, Y_prediction)*100
print("The test classification works with " + str(accuracy) + "% accuracy")
      
#classification precision score, metrics log loss
from sklearn.metrics import precision_score
from sklearn.metrics import log_loss

precision = precision_score(Y_test, Y_prediction, average = "weighted")*100
loss = log_loss(Y_test, Y_prediction)*100
print("Precision: " + str(precision))
print("Loss: " + str(loss))

#time program took to run
print(str(time.time() - t0) + " seconds wall time.")


