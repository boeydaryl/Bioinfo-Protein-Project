import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
import SVMInput




def opener(filename):
    AAlist, Statelist = SVMInput.threelineparser(filename)
    return AAlist, Statelist


def SVMscript1():
    for windowsize in range(1, 3, 2):
        Encoded_seq, Encoded_state = SVMInput.windowmaker_encoder(AAlist, Statelist, windowsize)
        print(Encoded_state)
        AA_array = np.array(Encoded_seq)
        state_array = np.array(Encoded_state)
        X_train, Y_train = AA_array, state_array
        C_range = [1, 5, 10]
        g_range = [0.001, 0.01]
        param = {'C' : C_range, 'gamma' : g_range}
        clf = GridSearchCV(SVC(), param, n_jobs=1, cv=3, verbose=2, error_score=np.NaN, return_train_score=False)
        clf.fit(X_train, Y_train)
        df = pd.DataFrame(clf.cv_results_)
        filename = str(windowsize) + '.csv'
        df.to_csv(filename, sep='\t', encoding='UTF-8')

    
    
    
    
    

if __name__ == "__main__":
    AAlist, Statelist = opener('../Datasets/testfilesize50.txt')
    SVMscript1()
