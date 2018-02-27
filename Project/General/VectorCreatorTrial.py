#Trial SVM classifier

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
#from sklearn.feature_extraction import DictVectorizer
#from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import OneHotEncoder
from numpy import array
from collections import deque
#from itertools import chain
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score



def threelineparser(filename, outputfilename):
    global datadict
    global df
    global AAlist
    global Statelist
    file1 = open(filename, 'r')
    datadict = dict()
    AAlist = []
    Statelist = []
    for x in file1:
        listAB = []
        if ">" in x:
            key = x.replace("\n", "")
        elif x.isupper() and ">" not in x:
            A = x.replace("\n", "")
            AAlist.append(A)
        elif x.islower() and ">" not in x:
            B = x.replace("\n", "")
            Statelist.append(B)
            datadict[key] = [A, B]
    df = pd.DataFrame.from_dict(data=datadict, orient='index')
    return AAlist
    return Statelist
    return datadict
    

def windowmaker_encoder(A, S, windowsize):
    global encoded_seq_list
    global encoded_state_list
    encoded_seq_list = []
    encoded_state_list = []
    AADict = {'A':[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'R':[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'N':[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'D':[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, ], 'C':[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Q':[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'E':[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'G':[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'H':[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'I':[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'L':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'K':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 'M':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 'F':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 'P':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 'S':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 'T':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 'W':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 'Y':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 'V':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 'B':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    StateDict = {'e':[0], 'b':[1]}
    pad = windowsize//2
    #window generator, storing a list of encoded sequences [list of vectors per frame]
    for seq in A:
        windowlist = []
        encoded_seq = []
        state_list = []
        print(seq)
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
        #list of combined vectors for each window, for 1 sequence
        for frame in windowlist:
            frame_list = [] #Combined vector for each window
            for AA in frame:
                if AA in AADict.keys():
                    frame_list.extend(AADict[AA])
                    frame_list_j = ''.join(map(str, frame_list))
            encoded_seq.append(frame_list_j)
        encoded_seq_list.append(encoded_seq)
    #print(frame_list_j)
    #print(encoded_seq)
    #print(encoded_seq_list)
    for states in S:
        state_list = []
        for state in states:
            if state in StateDict.keys():
                state_list.extend(StateDict[state])
                state_list_j = ''.join(map(str, state_list))
        encoded_state_list.append(state_list)
    #print(encoded_state_list)
    #print(state_list)
    return (encoded_seq_list, encoded_state_list)


def SVMscript(ESL1, ESL2):
    seqcount = 0
    for element in ESL1:
        seqcount += 1
    statecount = 0
    for element in ESL2:
        statecount += 1
    assert seqcount == statecount
    AA_array = np.array(ESL1)
    state_array = np.array(ESL2)
    print(AA_array)
    print(state_array)
    x,y = AA_array[:-1], state_array[:-1]
    print(x,y)
    clf = SVC(gamma =0.001, kernel = 'linear', C=1.0).fit(x,y)
    
    
    

if __name__ == "__main__":
    threelineparser('/Users/daryl/Documents/Bioinfo-Protein-Project/Project/Datasets/testfile5.txt', 'fulloutput.csv')
    print(windowmaker_encoder(AAlist, Statelist, 3))
    SVMscript(encoded_seq_list, encoded_state_list)