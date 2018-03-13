#complete PSSM Parser

import numpy as np
import PSSMParser

def generator(filename):
    listofheaders, listoftopo = PSSMParser.ID_topo_caller(filename)
    #print(listoftopo)
    return listofheaders,listoftopo
    
def PSSMcaller():
    listoffiles = []
    listofarrays = []
    for headers in listofheaders:
        names = str(headers) + '_FASTA.txt.pssm'
        listoffiles.append(names)
    #print(listoffiles)
    for name in listoffiles:
        PSSM_array = (np.genfromtxt('../Datasets/PSSM/' + name, skip_header = 3, skip_footer = 5, usecols = range(22,42), autostrip = True))/100
        listofarrays.append(PSSM_array)
    return listofarrays
    
    
def windowmaker(windowsize):
    multiseqwinlist = []
    padding = windowsize//2
    pad = np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    #print(len(pad))
    for seq in listofarrays:
        windowlist = []
        print(len(seq))
        for v in range(0, len(seq)):
            x_window =[]
            #print(seq[v])
            #print(len(seq[v]))
            if v <= 0:
                #x_window = []
                seq_window = seq[(v):(v+padding+1)]
                diff = windowsize-len(seq_window)
                for i in range(0, diff):
                    x_window.append(pad)
                #print(len(x_window))
                x_window.extend(seq_window)
                #print(x_window)
                #windowlist.append(x_window)
                #print(len(windowlist))
                #print(windowlist)
            elif v > 0 and v < padding:
                #x_window = []
                seq_window2 = seq[0:(v+padding+1)]
                diff = windowsize-len(seq_window2)
                for i in range(0, diff):
                    x_window.append(pad)
                #print(len(x_window))
                x_window.extend(seq_window2)
                #print(x_window)
                #windowlist.append(x_window)
                #print(len(windowlist))
            elif v >= padding:
                seq_window3 = seq[(v-padding):(v+padding+1)]
                if len(seq_window3) == windowsize:
                    #print(len(seq_window3))
                    x_window.extend(seq_window3)
                    #windowlist.append(seq_window3)
                    #print(windowlist)
                if len(seq_window3) < windowsize:
                    #x_window = []
                    diff = windowsize-len(seq_window3)
                    x_window.extend(seq_window3)
                    for i in range(0, diff):
                        x_window.append(pad)
            x_window = np.array(x_window)
            #print(len(x_window))
            x_window_f = x_window.flatten()
            windowlist.append(x_window_f)
        multiseqwinlist.extend(windowlist)
    #print(windowlist)
    windowarray = np.array(multiseqwinlist)
    print(windowarray.shape)
    #print(len(multiseqwinlist))
    
    
if __name__ == '__main__':
    listofheaders, listoftopo = generator('../Datasets/testfile3.txt')
    listofarrays = PSSMcaller()
    windowmaker(21)