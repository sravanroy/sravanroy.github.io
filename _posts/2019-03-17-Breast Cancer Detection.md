---
title: "Breast Cancer Detection"
date: 2018-12-29
tags: [machine learning, KNN, SVM, Classification]
header:
  image: "/images/breast_cancer/BCD.jpg"
excerpt: "Machine Learning, Classifiers, KNN, SVM"
mathjax: "true"
---

## *Implementation of clustering algorithms to predict breast cancer !*
----
* The dataset is retrieved directly from uci repository
+ SVM and KNN models were deployed to predict the cancer class as malign or benign   
_ The models were implemented in Python Jupyter notebook

```python
# Load all the required libraries

import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd
```

```python
# Load Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"

# get the column names of the dataset
names = ['id', 'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
       'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
       'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']
df = pd.read_csv(url, names=names)
```

```python
# Preprocess the data

df.replace('?',-99999, inplace=True) # ignore the blank data
print(df.axes)

df.drop(['id'], 1, inplace=True)
```

Lets look at the data in columns
```python
print(df.loc[10]) # looking at the 10th row

# Print the shape of the dataset
print(df.shape)

clump_thickness           1
uniform_cell_size         1
uniform_cell_shape        1
marginal_adhesion         1
single_epithelial_size    1
bare_nuclei               1
bland_chromatin           3
normal_nucleoli           1
mitoses                   1
class                     2
Name: 10, dtype: object
(699, 10)
```
```python
# Describe the dataset
print(df.describe())

 clump_thickness  uniform_cell_size  uniform_cell_shape  \
count       699.000000         699.000000          699.000000   
mean          4.417740           3.134478            3.207439   
std           2.815741           3.051459            2.971913   
min           1.000000           1.000000            1.000000   
25%           2.000000           1.000000            1.000000   
50%           4.000000           1.000000            1.000000   
75%           6.000000           5.000000            5.000000   
max          10.000000          10.000000           10.000000   

       marginal_adhesion  single_epithelial_size  bland_chromatin  \
count         699.000000              699.000000       699.000000   
mean            2.806867                3.216023         3.437768   
std             2.855379                2.214300         2.438364   
min             1.000000                1.000000         1.000000   
25%             1.000000                2.000000         2.000000   
50%             1.000000                2.000000         3.000000   
75%             4.000000                4.000000         5.000000   
max            10.000000               10.000000        10.000000   

       normal_nucleoli     mitoses       class  
count       699.000000  699.000000  699.000000  
mean          2.866953    1.589413    2.689557  
std           3.053634    1.715078    0.951273  
min           1.000000    1.000000    2.000000  
25%           1.000000    1.000000    2.000000  
50%           1.000000    1.000000    2.000000  
75%           4.000000    1.000000    4.000000  
max          10.000000   10.000000    4.000000  
```

```python
# Plot histograms for each variable

df.hist(figsize = (10, 10))
plt.show()
```

![alt]({{ site.url }}{{ site.baseurl }}/images/breast_cancer/hist.png)

```python
# Create scatter plot matrix

scatter_matrix(df, figsize = (18,18)) # to see the relationship between the variables
plt.show()
```
![alt]({{ site.url }}{{ site.baseurl }}/images/breast_cancer/scatter.png)

We dont have any strong relationships with the cancer class and hence linear models won't be a good fit

```python
# Create X and Y datasets for training and store the rest for validating the model
X = np.array(df.drop(['class'], 1)) # every variable except the class
y = np.array(df['class']) # only the class label which is to be predicted

# splitting the training and test data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# Testing Options
seed = 8 # to reproduce the same results
scoring = 'accuracy'

# Define models to train

models = []
models.append(('KNN', KNeighborsClassifier(n_neighbors = 5))) # add the KNN model with k=5
models.append(('SVM', SVC())) # add the SVM model

# evaluate each model in turn
results = []
names = []

for name, model in models: # run  KFold 10 times on train data on both models to choose the best one
    kfold = model_selection.KFold(n_splits=10, random_state = seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
	
	KNN: 0.973117 (0.018442)
    SVM: 0.951558 (0.028352)
```
This is the accuracy for the training data for both models

```python
# Make predictions on validation dataset

for name, model in models:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(name)
    print(accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))
    
# Accuracy - ratio of correctly predicted observations to the total observations. 
# Recall -  ratio of correctly predicted positive observations to the total predicted positive observations
# Precision - ratio of correctly predicted positive observations to the all observations in actual class - yes.
# F1 score - F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false 

KNN
0.9428571428571428
             precision    recall  f1-score   support

          2       0.94      0.96      0.95        83
          4       0.95      0.91      0.93        57

avg / total       0.94      0.94      0.94       140

SVM
0.9571428571428572
             precision    recall  f1-score   support

          2       1.00      0.93      0.96        83
          4       0.90      1.00      0.95        57

avg / total       0.96      0.96      0.96       140

clf = SVC()

clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1]])# randomly assign values to the variables(columns)
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(prediction)

0.9571428571428572
[2]
```
*The scope extends to using further clustering models in addition to KNN and SVM which might have a better accuracy of prediction*



