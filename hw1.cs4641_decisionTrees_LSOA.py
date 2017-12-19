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

cv = train_test_split(X, Y, test_size=.33, random_state= 20)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.33, random_state= 20)

clf = tree.DecisionTreeClassifier(criterion = "gini", splitter='random', min_samples_leaf = 10, max_depth = 3)

clf = clf.fit(X_train, Y_train)

train_prediction = clf.predict(X_train)
trainaccuracy = accuracy_score(train_prediction, Y_train)*100
print("The training accuracy for this is " +str(trainaccuracy))
#output
Y_prediction = clf.predict(X_test)
accuracy = accuracy_score(Y_test, Y_prediction)*100
print("The test classification works with " + str(accuracy) + "% accuracy without pruning")
#precision outcomes
from sklearn.metrics import precision_score
from sklearn.metrics import log_loss
precision = precision_score(Y_test, Y_prediction, average = "weighted")*100
loss = log_loss(Y_test, Y_prediction)*100
print("Precision: " + str(precision))
print("Loss: " + str(loss))

#Graph Viz
with open("decisionTree1.txt", 'w') as f:
    f = tree.export_graphviz(clf, out_file = f)
    
##SciKit Pruning Code (from Piazza)
#info at http://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py

def prune(tree, min_samples_leaf = 75000):
    if tree.min_samples_leaf >= min_samples_leaf:
        raise Exception('Tree already more pruned')
    else:
        tree.min_samples_leaf = min_samples_leaf

        tree = tree.tree_
        for i in range(tree.node_count):
            n_samples = tree.n_node_samples[i]
            if n_samples <= min_samples_leaf:
                tree.children_left[i]=-1
                tree.children_right[i]=-1
                
print(clf.min_samples_leaf)    
#Call pruning, plotting functions
prune(clf)
print(clf.min_samples_leaf)

#Final Score, time taken to run model
new_Y_prediction = clf.predict(X_test)
accuracy2 = accuracy_score(Y_test, new_Y_prediction)*100
print("This classification works with  " + str(accuracy2) + "% accuracy with pruning, as this algorithm implements pruning")

#Graph Viz o Pruned Graph
with open("decisionTree2.txt", 'w') as f:
    f = tree.export_graphviz(clf, out_file = f)

#time program took to run
print(str(time.time() - t0) + " seconds wall time.")

#Visualizations for model accuracy
#Learning Curve Estimator, Cross Validation
skplt.estimators.plot_learning_curve(clf, X, Y, title = "Learning Curve: Decision Trees")
plt.show()

