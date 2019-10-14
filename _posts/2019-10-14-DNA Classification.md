---
title: "DNA Classification"
date: 2019-10-14
tags: [machine learning, SVM, KNN, DNA]
header:
  image: "/images/dna.png"
excerpt: "Machine Learning, KNN, SVM, Neural nets"
mathjax: "true"
---

----
* The DNA data set is obtained from the UCI repository
+ Several Machine Learning models were implemented to predict the class of the gene (promoter and non-promoter)  
* Used 10 fold cross validation on train sets and studied different relevant accuracy metrics for determing the best model

```python
import numpy as np
import pandas as pd
import sys
import sklearn

```
## Importing dataset and preprocessing it

```python
# import uci molecular biology data set

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data'

names = ['Class', 'id', 'Sequence']
data = pd.read_csv(url , names = names)

# build the class dataset using a custom pandas dataframe

classes = data.loc[:,'Class']

# generate list of DNA sequences
sequences = list(data.loc[:,'Sequence'])
dataset = {}

# loop through the sequences and split into individual nucleotides
for i, seq in enumerate(sequences):
    
    #split and remove tabs
    nucl = list(seq)
    nucl = [x for x in nucl if x!= '\t']
    
    # append class asssignment
    nucl.append(classes[i])
    
    # add to dataset
    dataset[i] = nucl

dframe = pd.DataFrame(dataset)

# transpose the DataFrame
df = dframe.transpose()

# rename the last column to class
df.rename(columns = {57:'Class'}, inplace = True)

# record value counts for each sequence

series = []
for name in df.columns:
    series.append(df[name].value_counts())

info = pd.DataFrame(series)
details = info.transpose()

# switch to numerical data using pd.getdummies()

numerical_df = pd.get_dummies(df)

# remove one of the class columns and rename to simply 'Class'

df = numerical_df.drop(columns=['Class_-'])

df.rename(columns = {'Class_+':'Class'}, inplace = True)

```

## Building Machine Learning algorithms

 
```python

# import the algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

from sklearn import model_selection

# create X and Y datasets for training

X = np.array(df.drop(['Class'],1))
y = np.array(df['Class'])

# define a seed for reproducibility
seed = 1

# split the data into training and testing datasets

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size = 0.25, random_state=seed)

# define scoring method
scoring = 'accuracy'

# Define models to train
names = ["Nearest Neighbors", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "SVM Linear", "SVM RBF", "SVM Sigmoid"]


classifiers = [
    
    KNeighborsClassifier(n_neighbors = 3),
    GaussianProcessClassifier(1*RBF(1)),
    DecisionTreeClassifier(max_depth = 5),
    RandomForestClassifier(max_depth = 5,  n_estimators =10, max_features =1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    SVC(kernel = 'linear'),
    SVC(kernel = 'rbf'),
    SVC(kernel = 'sigmoid')
]

models = zip(names, classifiers)

# evaluate each models in turn

results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits = 10, random_state =seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv = kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = '{0}: {1} ({2})'.format(name, cv_results.mean(), cv_results.std())
    print(msg)
    print("Validating on test data set")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Validation Accuracy")
    print(round(accuracy_score(y_test, predictions),2))
    print("Confusion Matrix/n")
    print(classification_report(y_test, predictions))


```
## Results

```python
Nearest Neighbors: 0.8232142857142858 (0.11390841738440759)
Validating on test data set
Validation Accuracy
0.78
Confusion Matrix/n
              precision    recall  f1-score   support

           0       1.00      0.65      0.79        17
           1       0.62      1.00      0.77        10

    accuracy                           0.78        27
   macro avg       0.81      0.82      0.78        27
weighted avg       0.86      0.78      0.78        27

Gaussian Process: 0.8732142857142857 (0.05615780426255853)
Validating on test data set
Validation Accuracy
0.89
Confusion Matrix/n
              precision    recall  f1-score   support

           0       1.00      0.82      0.90        17
           1       0.77      1.00      0.87        10

    accuracy                           0.89        27
   macro avg       0.88      0.91      0.89        27
weighted avg       0.91      0.89      0.89        27

Decision Tree: 0.7107142857142856 (0.15685850897141249)
Validating on test data set
Validation Accuracy
0.74
Confusion Matrix/n
              precision    recall  f1-score   support

           0       1.00      0.59      0.74        17
           1       0.59      1.00      0.74        10

    accuracy                           0.74        27
   macro avg       0.79      0.79      0.74        27
weighted avg       0.85      0.74      0.74        27

Random Forest: 0.6178571428571429 (0.12142857142857144)
Validating on test data set
Validation Accuracy
0.48
Confusion Matrix/n
              precision    recall  f1-score   support

           0       0.71      0.29      0.42        17
           1       0.40      0.80      0.53        10

    accuracy                           0.48        27
   macro avg       0.56      0.55      0.47        27
weighted avg       0.60      0.48      0.46        27


Neural Net: 0.875 (0.09682458365518543)
Validating on test data set
Validation Accuracy
0.93
Confusion Matrix/n
              precision    recall  f1-score   support

           0       1.00      0.88      0.94        17
           1       0.83      1.00      0.91        10

    accuracy                           0.93        27
   macro avg       0.92      0.94      0.92        27
weighted avg       0.94      0.93      0.93        27


AdaBoost: 0.925 (0.11456439237389601)
Validating on test data set
Validation Accuracy
0.85
Confusion Matrix/n
              precision    recall  f1-score   support

           0       1.00      0.76      0.87        17
           1       0.71      1.00      0.83        10

    accuracy                           0.85        27
   macro avg       0.86      0.88      0.85        27
weighted avg       0.89      0.85      0.85        27

Naive Bayes: 0.8375 (0.1375)
Validating on test data set
Validation Accuracy
0.93
Confusion Matrix/n
              precision    recall  f1-score   support

           0       1.00      0.88      0.94        17
           1       0.83      1.00      0.91        10

    accuracy                           0.93        27
   macro avg       0.92      0.94      0.92        27
weighted avg       0.94      0.93      0.93        27

SVM Linear: 0.85 (0.10897247358851683)
Validating on test data set
Validation Accuracy
0.96
Confusion Matrix/n
              precision    recall  f1-score   support

           0       1.00      0.94      0.97        17
           1       0.91      1.00      0.95        10

    accuracy                           0.96        27
   macro avg       0.95      0.97      0.96        27
weighted avg       0.97      0.96      0.96        27

SVM RBF: 0.7375 (0.11792476415070755)
Validating on test data set
Validation Accuracy
0.78
Confusion Matrix/n
              precision    recall  f1-score   support

           0       1.00      0.65      0.79        17
           1       0.62      1.00      0.77        10

    accuracy                           0.78        27
   macro avg       0.81      0.82      0.78        27
weighted avg       0.86      0.78      0.78        27

SVM Sigmoid: 0.5696428571428571 (0.1592092225048921)
Validating on test data set
Validation Accuracy
0.44
Confusion Matrix/n
              precision    recall  f1-score   support

           0       1.00      0.12      0.21        17
           1       0.40      1.00      0.57        10

    accuracy                           0.44        27
   macro avg       0.70      0.56      0.39        27
weighted avg       0.78      0.44      0.34        27

```

* The results can be interpreted well by considering the business problem at hand and determing the best metric to pick the model!

