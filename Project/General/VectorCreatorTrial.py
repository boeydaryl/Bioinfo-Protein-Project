#Trial SVM classifier

import pandas as pd
import numpy as np
from sklearn import svm
#from sklearn.feature_extraction import DictVectorizer
#from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import OneHotEncoder
from numpy import array
from collections import deque
#from itertools import chain



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
    

def windowmaker1(A, S, windowsize):
    global AAwindow
    global Statewindow
    test = []
    windowlist = []
    AAWindow = []
    AADict = {'A':[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'R':[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'N':[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'D':[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, ], 'C':[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Q':[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'E':[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'G':[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'H':[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'I':[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'L':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'K':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 'M':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 'F':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 'P':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 'S':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 'T':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 'W':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 'Y':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 'V':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 'O':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    StateDict = {'e':1, 'b':2}
    pad = windowsize//2
    print(A)
    for seq in A:
        print(len(seq))
        for AA in range(0, len(seq)):
            if AA <= 0:
                seq_window = seq[(AA):(AA+pad+1)]
                seq_window = (windowsize-len(seq_window))*'O'+ seq_window
                #print(seq_window)
                windowlist.append(seq_window)
            elif AA > 0 and AA < pad:
                seq_window2 = seq[0:(AA+pad+1)]
                seq_window2 = (windowsize-len(seq_window2))*'O'+ seq_window2
                #print(seq_window2)
                windowlist.append(seq_window2)
            elif AA >= pad:
                seq_window3 = seq[(AA-pad):(AA+pad+1)]
                #print(seq_window3)
                if len(seq_window3) == windowsize:
                    windowlist.append(seq_window3)
                if len(seq_window3) < windowsize:
                    seq_window3 = seq_window3 + (windowsize-len(seq_window3))*'O'
                    #print(seq_window3)
                    windowlist.append(seq_window3)
    print(windowlist)
    print(len(windowlist))




if __name__ == "__main__":
    threelineparser('/Users/daryl/Documents/Bioinfo-Protein-Project/Project/Datasets/testfile2.txt', 'fulloutput.csv')
    windowmaker1(AAlist, Statelist, 3)
    #classifier(datadict)