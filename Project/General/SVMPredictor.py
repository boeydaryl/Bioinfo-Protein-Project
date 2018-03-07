import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.externals import joblib


def Parser(testfilename, windowsize):
    file1 = open(testfilename, 'r')
    AAList = []
    Processed_AAList = []
    encoded_seq = []
    #print(file1)
    for x in file1:
        if '>' in x:
            print(x)
        if '>' not in x:
            AAList.extend(x.replace('\n', ''))
            CombinedAA = ''.join(AAList)
    Processed_AAList.append(CombinedAA)
    print(CombinedAA)
    AAlen = len(CombinedAA)
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
    'B':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    #Create sliding windows of same dimensions as SVMInput
    pad = windowsize//2
    for AA in range(0, len(CombinedAA)):
        windowlist = []
        #print(seq)
        #print(len(seq))
        if AA <= 0:
            seq_window = CombinedAA[(AA):(AA+pad+1)]
            seq_window = (windowsize-len(seq_window))*'B'+ seq_window
            windowlist.append(seq_window)
        elif AA > 0 and AA < pad:
            seq_window2 = CombinedAA[0:(AA+pad+1)]
            seq_window2 = (windowsize-len(seq_window2))*'B'+ seq_window2
            windowlist.append(seq_window2)
        elif AA >= pad:
            seq_window3 = CombinedAA[(AA-pad):(AA+pad+1)]
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
            encoded_seq.append(frame_list)
    #print(encoded_seq)
    return encoded_seq, AAlen
        
def SVMTest(model, E_seq, AAlen):
    StateDict = {0:'e', 1:'b'}
    Top = []
    clf = joblib.load(model)
    AA_array = np.array(E_seq)
    #print(AA_array.shape)
    #print(AAlen)
    assert AAlen == AA_array.shape[0]
    predicted = clf.predict(AA_array)
    #print(predicted)
    for x in predicted:
        if x in StateDict.keys():
            x = StateDict[x]
            Top.append(x)
    print(''.join(Top))
    assert len(Top) == AAlen

        
if __name__ == "__main__":
    encoded_seq, AAlen = Parser('/Users/daryl/Documents/Bioinfo-Protein-Project/Project/Datasets/5W0P_A.fasta.txt', 9)
    SVMTest('output_full.pkl', encoded_seq, AAlen)