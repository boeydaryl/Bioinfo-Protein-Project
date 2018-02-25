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
    file1 = open(filename, 'r')
    datadict = dict()
    for x in file1:
        listAB = []
        if ">" in x:
            key = x.replace("\n", "")
        elif x.isupper() and ">" not in x:
            A = x.replace("\n", "")
        elif x.islower() and ">" not in x:
            B = x.replace("\n", "")
            datadict[key] = [A, B]
    df = pd.DataFrame.from_dict(data=datadict, orient='index')
    #df.rename(columns={'0': 'AA', '1': 'state'})
    #df.to_csv(outputfilename)
    #df['changed'] = map(lambda x: x.replace("e", 1), df[1])
    #df[1] = df[1].astype(str).replace('e', 1)
    #print(datadict)
    return datadict
    

"""def windowmaker(D, windowsize):
    global AAwindow
    global Statewindow
    AAlist = []
    Statelist = []
    AAwindow = []
    Statewindow = []
    for x, y in D.items():
        for z in y:
            if z.isupper():
                AAlist.append(z)
            if z.islower():
                Statelist.append(z)
    for seq in AAlist:
        for AA in range(0, len(seq)-(windowsize-1)):
            AAwindow.append(seq[AA:AA+windowsize])
    for element in Statelist:
        for elementi in range(0, len(element)-(windowsize-1)):
            Statewindow.append(element[elementi:elementi+windowsize])
    for seq in AAlist1:
        distance = windowsize-len(seq)
        if len(seq) < windowsize:
            padded = distance*'O' + seq
            print(padded)
    for seq in AAlist:
        print(seq)
        for AA in range(0, len(seq)):
            if AA-windowsize <=0:
                distance = (-(AA-windowsize)-1)
                if distance >= 1:
                    print(distance)
                    Padded = distance*'O' + seq[0]
                    if len(Padded) < windowsize:
                        Padded = Padded + seq[distance]
                    AAwin.append(Padded)
            AAwindow.append(seq[AA:AA+windowsize])
    return AAwindow
    return Statewindow"""


def windowmaker1(D, windowsize):
    global AAwindow
    global Statewindow
    AAlist = []
    Statelist = []
    AAwindow = []
    Statewindow = []
    for x, y in D.items():
        for z in y:
            if z.isupper():
                AAlist.append(z)
            if z.islower():
                Statelist.append(z)
    for seq in AAlist:
        window = deque(maxlen=windowsize)
        AAlist1 = []
        for AA in seq:
            window.append(AA)
            window1 = ''.join(window)
            if len(window1) < windowsize:
                distance = windowsize - len(window1)
                padded = distance*'O' + window1
                AAwindow.append(padded)
            else:
                AAwindow.append(window1)

    for states in Statelist:
            window = deque(maxlen=windowsize)
            statelist1 = []
            for state in states:
                window.append(state)
                window1 = ''.join(window)
                if len(window1) < windowsize:
                    distance = windowsize - len(window1)
                    padded = distance*'0' + window1
                    Statewindow.append(padded)
                else:
                    Statewindow.append(window1)
    print(AAwindow)
    print(Statewindow)
    return AAwindow
    return Statewindow
    

def classifier(D):
    #le = LabelEncoder()
    #ohe = OneHotEncoder(sparse=False)
    Encoded_AAWindow = []
    encoded_features = []
    AADict = {'A':[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'R':[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'N':[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'D':[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, ], 'C':[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Q':[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'E':[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'G':[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'H':[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'I':[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'L':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'K':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 'M':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 'F':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 'P':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 'S':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 'T':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 'W':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 'Y':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 'V':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 'O':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    #AADict = {'A':'10000000000000000000', 'R':'01000000000000000000', 'N':'00100000000000000000', 'D':'00010000000000000000', 'C':'00001000000000000000', 'Q':'00000100000000000000', 'E':'00000010000000000000', 'G':'00000001000000000000', 'H':'00000000100000000000', 'I':'00000000010000000000', 'L':'00000000001000000000', 'K':'00000000000100000000', 'M':'00000000000010000000', 'F':'00000000000001000000', 'P':'00000000000000100000', 'S':'00000000000000010000', 'T':'00000000000000001000', 'W':'00000000000000000100', 'Y':'00000000000000000010', 'V':'00000000000000000001', 'O':'00000000000000000000'}
    StateDict = {'e':1, 'b':2, '0':0}
    #print(AADict)
    for seq in AAwindow:
        for AA in seq:
            if AA in AADict.keys():
                list1 = []
                #Encoded_AAWindow.append(list1)
                Encoded_AAWindow.append(AADict[AA])
    for states in Statewindow:
        for state in states:
            if state in StateDict.keys():
                list1 = []
                list1.append(StateDict[state])
        encoded_features.append(list1)
    print(Encoded_AAWindow)
    print(encoded_features)
    flat_encoded_AAWindow = (item for items in Encoded_AAWindow for item in items)
    data1 = np.array(Encoded_AAWindow)
    data2 = np.array(encoded_features)
    #print(flat_encoded_AAWindow)
    print(data1)
    print(data2)
    """data1 = array(AAwindow)
    AAwindow_le = le.fit_transform(data1)
    AAwindow_le = AAwindow_le.reshape(len(AAwindow_le), 1)
    AAwindow_ohe = ohe.fit_transform(AAwindow_le)
    print(AAwindow_ohe)
    data2 = array(Statewindow)
    Statewindow_le = le.fit_transform(data2)
    Statewindow_le = Statewindow_le.reshape(len(Statewindow_le), 1)
    Statewindow_ohe = ohe.fit_transform(Statewindow_le)
    print(Statewindow_ohe)"""





if __name__ == "__main__":
    threelineparser('/Users/daryl/Documents/Bioinfo-Protein-Project/Project/Datasets/testfile3.txt', 'fulloutput.csv')
    #windowmaker(datadict, 3)
    windowmaker1(datadict, 3)
    classifier(datadict)