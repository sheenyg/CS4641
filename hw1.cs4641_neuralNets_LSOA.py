#Sheena Ganju, CS 4641 HW 1
#Nueral network implementation using scikit learn,
#help from http://www.kdnuggets.com/2016/10/beginners-guide-neural-networks-python-scikit-learn.html/2


#import statements
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
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
clf = MLPClassifier(solver= 'sgd', alpha = 1e-5, warm_start= True, max_iter = 5000)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.33, random_state= 20)

#scale for multi-layer perceptron
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

clf.fit(X_train, Y_train)

#predict the training and test accuracy
train_prediction = clf.predict(X_train)
trainaccuracy = accuracy_score(train_prediction, Y_train)*100
print("The training accuracy for this is " +str(trainaccuracy))

Y_prediction = clf.predict(X_test)
accuracy = accuracy_score(Y_test, Y_prediction)*100
print("The test accuracy works with " + str(accuracy) + "% accuracy")

#precision outcomes
from sklearn.metrics import precision_score
from sklearn.metrics import log_loss
precision = precision_score(Y_test, Y_prediction, average = "weighted")*100
loss = log_loss(Y_test, Y_prediction)*100
print("Precision: " + str(precision))
print("Loss: " + str(loss))

#time program took to run
print(str(time.time() - t0) + " seconds wall time.")

#Learning Curve Estimator, Cross Validation
##skplt.estimators.plot_learning_curve(clf, X, Y, title = "Learning Curve: Decision Trees")
##plt.show()

#plotting different learning rates, code from :http://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_training_curves.html#
#sphx-glr-auto-examples-neural-networks-plot-mlp-training-curves-py
params = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,
           'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0,
           'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
          {'solver': 'adam', 'learning_rate_init': 0.01}]

labels = ["constant learning-rate", "constant with momentum",
          "constant with Nesterov's momentum",
          "inv-scaling learning-rate", "inv-scaling with momentum",
          "inv-scaling with Nesterov's momentum", "adam"]

plot_args = [{'c': 'red', 'linestyle': '-'},
             {'c': 'green', 'linestyle': '-'},
             {'c': 'blue', 'linestyle': '-'},
             {'c': 'red', 'linestyle': '--'},
             {'c': 'green', 'linestyle': '--'},
             {'c': 'blue', 'linestyle': '--'},
             {'c': 'black', 'linestyle': '-'}]


def plot_on_dataset(X, y, name):
    # for each dataset, plot learning for each learning strategy
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(name)
    X = MinMaxScaler().fit_transform(X)
    mlps = []
    for label, param in zip(labels, params):
        max_iter = 150
        print("training: %s" % label)
        mlp = MLPClassifier(verbose=0, random_state=0,
                            max_iter=max_iter, **param)
        mlp.fit(X, y)
        mlps.append(mlp)
        print("Training set score: %f" % mlp.score(X, y))
        print("Training set loss: %f" % mlp.loss_)
    for mlp, label, args in zip(mlps, labels, plot_args):
            ax.plot(mlp.loss_curve_, label=label, **args)


fig, axes = plt.subplots(2, 2, figsize=(15, 10))
plot_on_dataset(X, Y, "Neural Nets")

plt.show()

