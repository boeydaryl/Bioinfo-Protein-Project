#complete PSSM Parser

import numpy as np
import PSSMParser
from sklearn import svm
from sklearn.svm import SVC
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.externals import joblib
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


def generator(filename):
    listofheaders, listoftopo = PSSMParser.ID_topo_caller(filename)
    #print(listoftopo)
    return listofheaders,listoftopo
    
def PSSMcaller(listofheaders):
    listoffiles = []
    listofarrays = []
    for headers in listofheaders:
        names = str(headers) + '_FASTA.txt.pssm'
        listoffiles.append(names)
    #print(listoffiles)
    for name in listoffiles:
        PSSM_array = (np.genfromtxt('../Datasets/PSSM/' + name, skip_header = 3, skip_footer = 5, usecols = range(22,42), autostrip = True))/100
        listofarrays.append(PSSM_array)
    return listofarrays
    
    
def PSSMwindowmaker(windowsize, listofarrays):
    multiseqwinlist = []
    padding = windowsize//2
    pad = np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    #print(len(pad))
    for seq in listofarrays:
        windowlist = []
        #print(len(seq))
        for v in range(0, len(seq)):
            x_window =[]
            #print(seq[v])
            #print(len(seq[v]))
            if v <= 0:
                #x_window = []
                seq_window = seq[(v):(v+padding+1)]
                diff = windowsize-len(seq_window)
                for i in range(0, diff):
                    x_window.append(pad)
                #print(len(x_window))
                x_window.extend(seq_window)
                #print(x_window)
                #windowlist.append(x_window)
                #print(len(windowlist))
                #print(windowlist)
            elif v > 0 and v < padding:
                #x_window = []
                seq_window2 = seq[0:(v+padding+1)]
                diff = windowsize-len(seq_window2)
                for i in range(0, diff):
                    x_window.append(pad)
                #print(len(x_window))
                x_window.extend(seq_window2)
                #print(x_window)
                #windowlist.append(x_window)
                #print(len(windowlist))
            elif v >= padding:
                seq_window3 = seq[(v-padding):(v+padding+1)]
                if len(seq_window3) == windowsize:
                    #print(len(seq_window3))
                    x_window.extend(seq_window3)
                    #windowlist.append(seq_window3)
                    #print(windowlist)
                if len(seq_window3) < windowsize:
                    #x_window = []
                    diff = windowsize-len(seq_window3)
                    x_window.extend(seq_window3)
                    for i in range(0, diff):
                        x_window.append(pad)
            x_window = np.array(x_window)
            #print(len(x_window))
            x_window_f = x_window.flatten()
            windowlist.append(x_window_f)
        multiseqwinlist.extend(windowlist)
    #print(windowlist)
    windowarray = np.array(multiseqwinlist)
    #print(windowarray.shape)
    #print(len(multiseqwinlist))
    return windowarray
    
def Topowindowmaker(windowsize):
    TopoDict = {'e': [0], 'b': [1]}
    statelist = []
    print(len(listoftopo))
    for states in listoftopo:
        for state in states:
            if state in TopoDict.keys():
                statelist.extend(TopoDict[state])
    statearray = np.array(statelist)
    #print(statearray.shape)
    return statearray
    
def PSSM_SVM(cvfold):
    x, y = windowarray, statearray
    print(x.shape)
    clf = SVC(gamma = 0.01, C = 5.0, kernel = 'rbf')
    scores = cross_val_score(clf, x, y, cv=cvfold, scoring='Accuracy')
    mean_scores = np.mean(scores)
    print(mean_scores)
    #y_predicted = cross_val_predict(clf, x, y, cv=cvfold)
    #conf_matrix = confusion_matrix(y, y_predicted)
    #print(conf_matrix)
    
    
def PSSM_split():
    x, y = windowarray,statearray
    x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.33, random_state=10)
    clf = SVC(gamma = 0.01, C = 5.0, kernel = 'rbf')
    clf.fit(x_train, y_train)
    y_predicted = clf.predict(x_test)
    MatCorr = matthews_corrcoef(y_test, y_predicted)
    print(MatCorr)

def PSSM_others():
    x, y = windowarray, statearray
    clf = tree.DecisionTreeClassifier(random_state=10)
    #clf = RandomForestClassifier(random_state=10)
    #x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.33, random_state=10)
    #clf = clf.fit(x, y_train)
    #y_predict = clf.predict(x_test)
    scores = cross_val_score(clf, x, y, cv=3, scoring = 'f1')
    meanscores = np.mean(scores)
    print(meanscores)
    
def PSSM_model(outputfilename):
    x, y = windowarray, statearray
    clf = SVC(gamma = 0.01, C = 5.0, kernel = 'rbf')
    PSSM_model = clf.fit(x, y)
    joblib.dump(PSSM_model, outputfilename)
    
if __name__ == '__main__':
    listofheaders, listoftopo = generator('../Datasets/testfilesize50.txt')
    listofarrays = PSSMcaller(listofheaders)
    windowarray = PSSMwindowmaker(21, listofarrays)
    statearray = Topowindowmaker(21)
    #PSSM_SVM(3)
    #PSSM_split()
    PSSM_others()
    #PSSM_model('../Datasets/PSSMoutput.pkl')
