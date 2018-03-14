import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import SVMInput
import SVMPredictor
import PSSMComplete

#Final Predictor Script

############# AA Parser #############
def Parser(filename):
    header_list, AAList, Statelist, seq_len = SVMPredictor.Parser(filename)
    return header_list, AAList, Statelist, seq_len
    
############ AA Window ##############
def WindowMaker(A, S, windowsize):
    encoded_seq, encoded_state = SVMInput.windowmaker_encoder(AAList, Statelist, 21)
    return encoded_seq, encoded_state

######### AA Seq Model ##########
def AA_Seq_Model(model, encoded_seq, encoded_state):
    clf = joblib.load(model)
    x, y = encoded_seq, encoded_state
    y_predict = clf.predict(x)
    #print(y_predict)
    MatCorr = matthews_corrcoef(y, y_predict)
    print(MatCorr)
    #accuracy = accuracy_score(y, y_predict)
    #f1score = f1_score(y, y_predict, average = 'macro')
    #print('Accuracy = ' + str(accuracy), 'F1_score = ' + str(f1score))
    

########### PSSM Parser ##########
def generator(filename):
    listofheaders, listoftopo = PSSMComplete.generator(filename)
    return listofheaders, listoftopo
    
def PSSMCaller(listofheaders):
    listoffiles = []
    listofarrays = []
    for headers in listofheaders:
        names = str(headers) + '_FASTA.txt.pssm'
        listoffiles.append(names)
    for name in listoffiles:
        PSSM_array = (np.genfromtxt('../Datasets/50Extra/PSSM/' + name, skip_header = 3, skip_footer = 5, usecols = range(22,42), autostrip = True))/100
        listofarrays.append(PSSM_array)
    return listofarrays
    
def PSSMWindow(windowsize, listofarrays):
    windowarray = PSSMComplete.PSSMwindowmaker(windowsize, listofarrays)
    return windowarray
    
######## PSSM Model #############
def PSSM_Model(model, encoded_seq, encoded_state):
    clf = joblib.load(model)
    x = windowarray
    y = np.array(encoded_state)
    y_predict = clf.predict(x)
    MatCorr = matthews_corrcoef(y, y_predict)
    print(MatCorr)
    #accuracy = accuracy_score(y, y_predict)
    #f1score = f1_score(y, y_predict, average = 'macro')
    #print('Accuracy = ' + str(accuracy), 'F1_score = ' + str(f1score))
    
    
if __name__ == '__main__':
    Header_list, AAList, Statelist, seq_len = Parser('../Datasets/50Extra.txt')
    encoded_seq, encoded_state = WindowMaker(AAList, Statelist, 21)
    #AA_Seq_Model('../Datasets/output_50.pkl', encoded_seq, encoded_state)
    listofheaders, listoftopo = generator('../Datasets/50Extra.txt')
    listofarrays = PSSMCaller(listofheaders)
    windowarray = PSSMWindow(21, listofarrays)
    PSSM_Model('../Datasets/PSSMoutput.pkl', encoded_seq, encoded_state)