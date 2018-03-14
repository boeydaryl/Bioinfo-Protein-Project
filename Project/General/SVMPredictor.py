import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.externals import joblib
import SVMInput

############# Parser for regular FASTA files###########

def Parser(testfilename):
    file1 = open(testfilename, 'r')
    seq_list = []
    header_list = []
    seq_len = []
    for x in file1:
        if '>' in x:
            header_list.append(x)
            #print(x)
        elif '>' not in x and x.isupper():
            #print(x)
            seq_list.append(x.replace('\n', ''))
    #print(seq_list)
    for x in seq_list:
        seq_len.append(len(x))
    #print(seq_len)
    #AAlen = len(seq)
    file1.close()
    return header_list, seq_list, seq_len
    
############## Window making and Encoding ####################
    
def Encoder(seq_list, windowsize):
    encoded_seq = []
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
    for seq in seq_list:
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
            encoded_seq.append(frame_list)
    #print(encoded_seq)
    return encoded_seq
        
        
################# SVM Predictor on new sequences#############
        
def SVMTest(model, E_seq, Seq_len, header_list, seq_list, filename):
    StateDict = {0:'e', 1:'b'}
    Topo = []
    TopoList = []
    Corrected_seq_len = []
    clf = joblib.load(model)
    AA_array = np.array(E_seq)
    predicted = clf.predict(AA_array)
    for x in predicted:
        if x in StateDict.keys():
            x = StateDict[x]
            Topo.append(x)
    #for loop to extract states based on sequence length
    bigsum = 0
    writefile = open(filename, 'w')
    for n in seq_len:
        bigsum += n
        Corrected_seq_len.append(bigsum)
    for n in range(len(Corrected_seq_len)):
        list1 = []
        if n < 1:
            x = Topo[0:Corrected_seq_len[0]]
            y = ''.join(x)
            print(header_list[n])
            print(seq_list[n])
            print(y)
            writefile.write(header_list[n])
            writefile.write(seq_list[n] + '\n')
            writefile.write(y + '\n')
        else:
            x1 = Topo[Corrected_seq_len[n-1]:Corrected_seq_len[n]]
            y = ''.join(x1)
            print(header_list[n])
            print(seq_list[n])
            print(y)
            writefile.write(header_list[n])
            writefile.write(seq_list[n] + '\n')
            writefile.write(y + '\n')
        TopoList.append(y)
            
        
if __name__ == "__main__":
    header_list, seq_list, seq_len = Parser('../Datasets/5W0P_A.fasta.txt')
    encoded_seq = Encoder(seq_list, 21)
    SVMTest('../Datasets/output_50.pkl', encoded_seq, seq_len, header_list, seq_list, '../Datasets/Predicted')
