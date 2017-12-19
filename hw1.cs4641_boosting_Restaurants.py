#Sheena Ganju, CS 4641 HW 1
#Decision Trees

#code uses scikit-learn decision trees, code from cstrelioff on github and
#http://scikit-learn.org/stable/modules/tree.html, skeleton code from
#http://dataaspirant.com/2017/02/01/decision-tree-algorithm-python-with-scikit-learn/

#import statements
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
import scikitplot as skplt
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import validation_curve
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import zero_one_loss

#Read data in using pandas

#read in data
#Read data in using pandas
trainDataSet = pd.read_csv("geoplaces2.csv", sep = ',', header = None, low_memory = False)

#encode text data to integers using getDummies
traindata = pd.get_dummies(trainDataSet)

# Create decision Tree using major_category, month, year, to predict violent or not 
# train split uses default gini node, split using train_test_split

X = traindata.values[1:, 1:]
Y = traindata.values[1:,0]

#start timer
t0= time.clock()

cv = train_test_split(X, Y, test_size=.33, random_state= 20)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.33, random_state= 20)

#fitting classifiers
clf = AdaBoostClassifier(tree.DecisionTreeClassifier(criterion = "gini"), n_estimators = 100)
clf = clf.fit(X_train, Y_train)

train_prediction = clf.predict(X_train)
trainaccuracy = accuracy_score(train_prediction, Y_train)*100
print("The training accuracy for this is " +str(trainaccuracy))
#output
Y_prediction = clf.predict(X_test)
accuracy = accuracy_score(Y_test, Y_prediction)*100
print("The test classification works with " + str(accuracy) + "% accuracy without pruning")

#classification precision score, metrics log loss
from sklearn.metrics import precision_score
from sklearn.metrics import log_loss

precision = precision_score(Y_test, Y_prediction, average = "weighted")*100
loss = log_loss(Y_test, Y_prediction)*100
print("Precision: " + str(precision))
print("Loss: " + str(loss))

#time program took to run
print(str(time.time() - t0) + " seconds wall time.")

#Visualizations for model accuracy
#Learning Curve Estimator, Cross Validation

##skplt.estimators.plot_learning_curve(clf, X, Y, title = "Learning Curve: Boosted Decision Trees")
##plt.show()

clf2 = tree.DecisionTreeClassifier(criterion ="gini")
clf2.fit(X_train, Y_train)
Y1 = clf2.predict(X_test)

#estimators vs error rate plot (copied from http://scikit-learn.org/stable/auto
#_examples/ensemble/plot_adaboost_hastie_10_2.html#sphx-glr-auto-examples-ensemb
#le-plot-adaboost-hastie-10-2-py
    
dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
dt_stump.fit(X_train, Y_train)
dt_stump_err = 1.0 - dt_stump.score(X_test, Y_test)
n_estimators = 2000

dt = DecisionTreeClassifier(max_depth=9, min_samples_leaf=1)
dt.fit(X_train, Y_train)
dt_err = 1.0 - dt.score(X_test, Y_test)

ada_real = AdaBoostClassifier(
    base_estimator=dt_stump,
    learning_rate=1,
    n_estimators= n_estimators,
    algorithm="SAMME.R")
ada_real.fit(X_train, Y_train)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot([1, n_estimators], [dt_stump_err] * 2, 'k-',
        label='Decision Stump Error')
ax.plot([1, n_estimators], [dt_err] * 2, 'k--',
        label='Decision Tree Error')

#graphic ada error 
ada_real_err = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_real.staged_predict(X_test)):
    ada_real_err[i] = zero_one_loss(y_pred, Y_test)

ada_real_err_train = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_real.staged_predict(X_train)):
    ada_real_err_train[i] = zero_one_loss(y_pred, Y_train)

ax.plot(np.arange(n_estimators) + 1, ada_real_err,
        label='Real AdaBoost Test Error',
        color='orange')
ax.plot(np.arange(n_estimators) + 1, ada_real_err_train,
        label='Real AdaBoost Train Error',
        color='green')

ax.set_ylim((0.0, .10))
ax.set_xlabel('n_estimators')
ax.set_ylabel('error rate')

leg = ax.legend(loc='upper right', fancybox=True)
leg.get_frame().set_alpha(0.7)

plt.show()
