#import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib


########### Parser#############################

def threelineparser(filename):
    file1 = open(filename, 'r')
    datadict = dict()
    AAlist = []
    Statelist = []
    AAcount = 0
    for x in file1:
        listAB = []
        if ">" in x:
            key = x.replace("\n", "")
            AAcount += 1
        elif x.isupper() and ">" not in x:
            A = x.replace("\n", "")
            AAlist.append(A)
        elif x.islower() and ">" not in x:
            B = x.replace("\n", "")
            Statelist.append(B)
            datadict[key] = [A, B]
    #df = pd.DataFrame.from_dict(data=datadict, orient='index')
    print(AAcount)
    file1.close()
    return AAlist, Statelist

############ Sliding Window and Encoding##########

def windowmaker_encoder(A, S, windowsize):
    encoded_seq = []
    encoded_state = []
    AADict = {
    'A':[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    'R':[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    'N':[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    'D':[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    'C':[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    'Q':[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    'E':[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    'G':[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    'H':[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    'I':[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    'L':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    'K':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
    'M':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
    'F':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 
    'P':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
    'S':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 
    'T':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
    'W':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 
    'Y':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
    'V':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
    'B':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]} #artificial amino acid
    StateDict = {'e':[0], 'b':[1]}
    pad = windowsize//2
    #window generator, storing a list of encoded sequences [list of vectors per frame]
    for seq in A:
        windowlist = []
        #print(seq)
        #print(len(seq))
        for AA in range(0, len(seq)):
            if AA <= 0:
                seq_window = seq[(AA):(AA+pad+1)]
                seq_window = (windowsize-len(seq_window))*'B'+ seq_window
                windowlist.append(seq_window)
            elif AA > 0 and AA < pad:
                seq_window2 = seq[0:(AA+pad+1)]
                seq_window2 = (windowsize-len(seq_window2))*'B'+ seq_window2
                windowlist.append(seq_window2)
            elif AA >= pad:
                seq_window3 = seq[(AA-pad):(AA+pad+1)]
                if len(seq_window3) == windowsize:
                    windowlist.append(seq_window3)
                if len(seq_window3) < windowsize:
                    seq_window3 = seq_window3 + (windowsize-len(seq_window3))*'B'
                    #print(seq_window3)
                    windowlist.append(seq_window3)
        #print(windowlist)
        for frame in windowlist:
            frame_list = [] #Combined vector for each window
            for AA in frame:
                if AA in AADict.keys():
                    frame_list.extend(AADict[AA])
                    #print(len(frame_list))
            assert len(frame_list) == windowsize*20
            encoded_seq.append(frame_list)#list of combined vectors for each window
    #print(len(frame_list))
    #print(encoded_seq)
    #print(len(encoded_seq))
    for states in S:
        for state in states:
            if state in StateDict.keys():
                encoded_state.extend(StateDict[state])
    return encoded_seq, encoded_state

################# SVM Cross Validator and Model Fitter###########

def SVMscript(E_seq, E_state, cvfold, filename):
    seqcount = 0
    for element in E_seq:
        seqcount += 1
    statecount = 0
    for element in E_state:
        statecount += 1
    assert seqcount == statecount
    #print(seqcount)
    AA_array = np.array(E_seq)
    state_array = np.array(E_state)
    #print(AA_array)
    #print(state_array)
    x, y = AA_array, state_array
    clf = SVC(gamma =0.01, kernel = 'rbf', C=5.0)
    #print(clf)
    #print(x.shape, y.shape)
    scores = cross_val_score(clf, x, y , cv=cvfold, scoring='accuracy')
    #print(scores)
    average_score = np.average(scores)
    print(average_score)
    #y_predicted = cross_val_predict(clf, x, y, cv=cvfold)
    #conf_matrix = confusion_matrix(y, y_predicted)
    #print(conf_matrix)
    #model = clf.fit(x, y)
    #print(scores)
    #joblib.dump(model, filename)
    #return average_score
    
    
    
    
    

if __name__ == "__main__":
    data_file = '../Datasets/317proteins.txt'
    AAlist, Statelist = threelineparser(data_file)
    encoded_seq, encoded_state = windowmaker_encoder(AAlist, Statelist, 21)
    print(SVMscript(encoded_seq, encoded_state, 3, '../Datasets/output_50.pkl'))
