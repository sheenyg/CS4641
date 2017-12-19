#Sheena Ganju, CS 4641 HW 1
#Nueral network implementation using scikit learn,
#help from http://www.kdnuggets.com/2016/10/beginners-guide-neural-networks-python-scikit-learn.html/2

from sklearn.metrics import accuracy_score
from sklearn import svm
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

#read in data
#Read data in using pandas
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

#set up classifier,, iterations controlled by warm_start= True and max_iter = 1

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.33, random_state= 20)

clf = svm.SVC(kernel = "rbf")
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


clf2 = svm.LinearSVC()
clf2.fit(X,Y)
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

#Plotting some kernels and stuff: http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html
def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# Take the first two features. We could avoid this by using a two-dim dataset
X = X[:, :2]

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))
models = (clf.fit(X, Y) for clf in models)

# title for the plots
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=Y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()
