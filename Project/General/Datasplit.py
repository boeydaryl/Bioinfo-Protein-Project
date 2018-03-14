#Datasplit
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
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import SVMInput

def extractor(filename):
    AAList, StateList = SVMInput.threelineparser(filename)
    return AAList, StateList


def datasplit(windowsize):
    e_seq, e_state = SVMInput.windowmaker_encoder(AAList, StateList, windowsize)
    x = np.array(e_seq)
    y = np.array(e_state)
    #x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.33, random_state=10)
    #print(y_train)
    #SVC = clf(C=5, gamma=0.01, kernel='rbf')
    #clf.fit(x_train, y_train)
    #y_predicted = clf.predict(x_test)
    #print(y_predicted)
    #MatCorr = matthews_corrcoef(y_test, y_predicted)
    #print(MatCorr)
    return x, y
    
def decisiontree(windowsize):
    clf = tree.DecisionTreeClassifier(random_state=10)
    #x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.33, random_state=10)
    #clf = clf.fit(x, y_train)
    #y_predict = clf.predict(x_test)
    scores = cross_val_score(clf, x, y, cv=3, scoring = 'f1')
    meanscores = np.mean(scores)
    print(meanscores)
    
def RandomForest(windowsize):
    clf = RandomForestClassifier(random_state=10)
    #x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.33, random_state=10)
    #clf = clf.fit(x, y_train)
    #y_predict = clf.predict(x_test)
    scores = cross_val_score(clf, x, y, cv=3, scoring = 'f1')
    meanscores = np.mean(scores)
    print(meanscores)
    




if __name__ == '__main__':
    AAList, StateList = extractor('../Datasets/testfilesize50.txt')
    x, y = datasplit(9)
    decisiontree(9)
    RandomForest(9)
