#Randomforest

import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
import SVMInput

def extractor(filename):
    AAList, StateList = SVMInput.threelineparser(filename)
    return AAList, StateList

    
def RanFor(windowsize):
    e_seq, e_state = SVMInput.windowmaker_encoder(AAList, StateList, windowsize)
    x = np.array(e_seq)
    y = np.array(e_state)
    seed = 10
    trees = 100
    kfold = KFold(n_splits=3, random_state=seed)
    clf = RandomForestClassifier(n_estimators=trees, max_features=3)
    RFResults = cross_val_score(clf, x, y, cv=kfold, scoring='average_precision')
    mean_RFResults = RFResults.mean()
    print(mean_RFResults)

    
    
if __name__ == '__main__':
    AAList, StateList = extractor('../Datasets/testfilesize50.txt')
    RanFor(9)