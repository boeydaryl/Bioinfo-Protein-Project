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
    FrameWindow = []
    AADict = {'A':[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'R':[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'N':[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'D':[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, ], 'C':[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Q':[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'E':[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'G':[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'H':[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'I':[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'L':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'K':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 'M':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 'F':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 'P':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 'S':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 'T':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 'W':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 'Y':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 'V':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 'O':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    StateDict = {'e':1, 'b':2}
    pad = windowsize//2
    print(A)
    for seq in A:
        window = deque(maxlen=windowsize)
        for AA in seq:
            list1 = []
            window.append(AA)
            window1 = ''.join(window)
            print(window1)
            for i in window1:
                if i in AADict.keys():
                    #i = AADict[i]
                    list1.extend(AADict[i])
            AAWindow.append(list1)
    for i in AAWindow:
        if len(i) >= (pad+1)*20:
            total_length = windowsize*20
            padding = total_length - len(i)
            i = int(padding/20)*[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + i
            FrameWindow.append(i)
    print(len(FrameWindow))
    
    """for seq in A:
        for AA in range(0, len(seq)):
            if AA < pad:
                print(seq[AA])
                if AA <= 0:
                    seq_window1 = pad*'O' + seq[AA:AA+pad+1]
                    print(seq_window1)
                    windowlist.append(seq_window1)
                else:
                    seq_window2 = pad//2*'O' + seq[(AA-pad//2):(AA+pad+1)]
                    print(seq_window2)
                    windowlist.append(seq_window2)
            #if AA >= pad and AA <= len(seq)-pad-1:
            seq_window3 = seq[(AA-pad):(AA+pad+1)]
            windowlist.append(seq_window3)
            print(seq_window3)
                
            for i in seq_window3:
                if i in AADict.keys():
                    i = AADict[i]
                    FrameWindow = i * windowsize
                    #print(FrameWindow)
                    AAWindow.append(FrameWindow)
        for AA in range(len(seq)):
        if AA < windowsize:
            seqwindow = pad*'0' + seq[AA]
            print(seqwindow)"""




if __name__ == "__main__":
    threelineparser('/Users/daryl/Documents/Bioinfo-Protein-Project/Project/Datasets/testfile2.txt', 'fulloutput.csv')
    windowmaker1(AAlist, Statelist, 5)
    #classifier(datadict)