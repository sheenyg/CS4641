Sheena Ganju
CS 4641 HW 1 
Code Running Instrutions

1. This code uses the scikit framework for Python 3.5. 
Packages involved for files labelled "hw1" include
sckikit, 
numpy, 
pandas, 
scipy, 
matplotlib, 
pyplot,
and 
dependencies that require installation before the code is run. 

Packages invovled for files labelled "hw4" include
sklearn.cluster,
sklearn.mixture,
sklearn.decomposition,
sklearn.random_projection
sknn.ae

2. The datasets included are labelled "london_crime_by_lsoa.csv" and "geoplaces2.csv". They are read in for the user each time 
a user calls one of the Python functions so long as the directory is correct (i.e. they are in the working directory). 

3. There are 8 Python files that can be run individually for each requirement and each dataset,
all labelled with the format "hw1.cs4641"_topic_dataset. 
The topics are referenced as "boosting, decisionTrees, kNN, neuralNets, SVM".
The datasets are referenced as "LSOA" and "Restaurants". 

4. All charts displayed in the graphs are in the respective section of the code. Some graphs are commented out, as only one plot (plt.show())
command should be called at once for ease of use. 

5. For decision trees, two .txt files with .dot code are created in your directory each time the code is run. 
To convert these into a graph, go to http://www.webgraphviz.com/, and paste the code from the .txt file and 
press submit. 