#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd

#glass is the dataframe
glass = pd.read_csv(r"https://raw.githubusercontent.com/jbrownlee/Datasets/master/glass.csv")

df = pd.DataFrame(glass)
df.info()
data = df.values

#All rows, 10th column
y = data[:, 9]
#All rows, columns 0 thru 7
X = data[:, 0:9]


# In[37]:


#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB(var_smoothing=1.8)

#Run cross validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
cv = RepeatedKFold(n_splits=10, n_repeats=10)
gnbScores = cross_val_score(gnb, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print("Gaussian Naive Bayes Accuracy: %0.2f (+/- %0.2f)" % (gnbScores.mean(), gnbScores.std() * 2))


# In[52]:


#SVM SVC
from sklearn import svm

clf = svm.SVC(C = 150, kernel='linear')
clf.fit(X, y)

#Run cross validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
cv = RepeatedKFold(n_splits=10, n_repeats=10)
svmScores = cross_val_score(clf, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print("Support Vector Machine Accuracy: %0.2f (+/- %0.2f)" % (svmScores.mean(), svmScores.std() * 2))


# In[68]:


#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(max_depth = 10, random_state = 150)
dtc = dtc.fit(X, y)

#Run cross validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
cv = RepeatedKFold(n_splits=10, n_repeats=10)
dtcScores = cross_val_score(dtc, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print("Decision Tree Classifier Accuracy: %0.2f (+/- %0.2f)" % (dtcScores.mean(), dtcScores.std() * 2))


# In[76]:


#KNN Classifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn.fit(X, y)

#Run cross validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
cv = RepeatedKFold(n_splits=10, n_repeats=10)
knnScores = cross_val_score(knn, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print("K Nearest Neighbors Accuracy: %0.2f (+/- %0.2f)" % (knnScores.mean(), knnScores.std() * 2))


# In[83]:


#AdaBoost Ensemble Classifier
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(n_estimators = 30)

#Run cross validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
cv = RepeatedKFold(n_splits=10, n_repeats=10)
adaScores = cross_val_score(ada, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print("AdaBoost Accuracy: %0.2f (+/- %0.2f)" % (adaScores.mean(), adaScores.std() * 2))


# In[84]:


#Compare Classifier Differences

#Gaussian Naive Bayes vs all other classsifiers
scoreDiffs = gnbScores - svmScores
print("GNB vs SVM -> Accuracy: %0.2f (+/- %0.2f)" % (scoreDiffs.mean(), scoreDiffs.std() * 2))
scoreDiffs = gnbScores - dtcScores
print("GNB vs DTC -> Accuracy: %0.2f (+/- %0.2f)" % (scoreDiffs.mean(), scoreDiffs.std() * 2))
scoreDiffs = gnbScores - knnScores
print("GNB vs KNN -> Accuracy: %0.2f (+/- %0.2f)" % (scoreDiffs.mean(), scoreDiffs.std() * 2))
scoreDiffs = gnbScores - adaScores
print("GNB vs ADA -> Accuracy: %0.2f (+/- %0.2f)" % (scoreDiffs.mean(), scoreDiffs.std() * 2))

#Support Vector Machine vs dtc, knn, and ada
scoreDiffs = svmScores - dtcScores
print("SVM vs DTC -> Accuracy: %0.2f (+/- %0.2f)" % (scoreDiffs.mean(), scoreDiffs.std() * 2))
scoreDiffs = svmScores - knnScores
print("SVM vs KNN -> Accuracy: %0.2f (+/- %0.2f)" % (scoreDiffs.mean(), scoreDiffs.std() * 2))
scoreDiffs = svmScores - adaScores
print("SVM vs ADA -> Accuracy: %0.2f (+/- %0.2f)" % (scoreDiffs.mean(), scoreDiffs.std() * 2))

#Decision Tree vs knn and ada
scoreDiffs = dtcScores - knnScores
print("DTC vs KNN -> Accuracy: %0.2f (+/- %0.2f)" % (scoreDiffs.mean(), scoreDiffs.std() * 2))
scoreDiffs = dtcScores - adaScores
print("DTC vs Ada -> Accuracy: %0.2f (+/- %0.2f)" % (scoreDiffs.mean(), scoreDiffs.std() * 2))

#K Neaster Neighbors vs ada
scoreDiffs = knnScores - adaScores
print("KNN vs ADA -> Accuracy: %0.2f (+/- %0.2f)" % (scoreDiffs.mean(), scoreDiffs.std() * 2))


# In[ ]:




