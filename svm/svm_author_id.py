#!/usr/bin/python
""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

## Take a subset to increase training speed
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

#########################################################
### your code goes here ###
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#clf1 = SVC(kernel = "rbf", C = 10)
#clf2 = SVC(kernel = "rbf", C = 100)
#clf3 = SVC(kernel = "rbf", C = 1000)
clf4 = SVC(kernel = "rbf", C = 10000)
t0 = time()
#clf_fit1 = clf1.fit(features_train,labels_train)
#clf_fit2 = clf2.fit(features_train,labels_train)
#clf_fit3 = clf3.fit(features_train,labels_train)
clf_fit4 = clf4.fit(features_train,labels_train)
print "training time:", round(time()-t0, 3), "s"
t1 = time()
#predict1 = clf1.predict(features_test)
#predict2 = clf2.predict(features_test)
#predict3 = clf3.predict(features_test)
predict4 = clf4.predict(features_test)

print "prediction time:", round(time()-t1, 3), "s"
#acc1 = accuracy_score(predict1, labels_test)
#acc2 = accuracy_score(predict2, labels_test)
#acc3 = accuracy_score(predict3, labels_test)
acc4 = accuracy_score(predict4, labels_test)
#print acc1
#print acc2
#print acc3
print acc4
#print predict4[10]
#print predict4[26]
#print predict4[50]

print sum(predict4)
#########################################################


