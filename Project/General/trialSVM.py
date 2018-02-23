#Trial SVM classifier

import pandas as pd
import numpy as np
from sklearn import svm
#from sklearn.feature_extraction import DictVectorizer
#from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import OneHotEncoder
from numpy import array



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
    

def windowmaker(D, windowsize):
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
    return AAwindow
    return Statewindow


def classifier(D):
    #le = LabelEncoder()
    #ohe = OneHotEncoder(sparse=False)
    Encoded_AAWindow = []
    Features = []
    AADict = {'A':['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0'], 'R':['0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0'], 'N':['0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0'], 'D':['0', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0'], 'C':['0', '0', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0'], 'Q':['0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0'], 'E':['0', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0'], 'G':['0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0'], 'H':['0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0'], 'I':['0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0'], 'L':['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0'], 'K':['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0'], 'M':['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0'], 'F':['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '0', '0'], 'P':['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '0'], 'S':['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0'], 'T':['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0'], 'W':['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0'], 'Y':['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0'], 'V':['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1'], 'O':['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']}
    #print(AADict)
    #print(AAwindow)
    for seq in AAwindow:
        list1 = []
        for AA in seq:
            if AA in AADict.keys():
                list1.append(AADict[AA])
        Encoded_AAWindow.append(list1)
    

    print(Encoded_AAWindow)
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
    threelineparser('/Users/daryl/Documents/Bioinfo-Protein-Project/Project/Datasets/testfile.txt', 'fulloutput.csv')
    windowmaker(datadict, 3)
    #classifier(datadict)