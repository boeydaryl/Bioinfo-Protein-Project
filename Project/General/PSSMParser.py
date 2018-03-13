#PSSM Parser
#To draw right side table for entire sequence

import numpy as np
import pandas as pd
from pandas import DataFrame
import SVMInput

def PSSM_ASCII_open(filename):
    PSSM_array = (np.genfromtxt(filename, skip_header = 3, skip_footer = 5, usecols = range(22,42), autostrip = True))/100
    #print(file1)
    df = pd.DataFrame(PSSM_array)
    #print(df)
    #filedf = str(filename) + '.csv'
    #df.to_csv(filedf, sep = '\t', encoding='UTF-8')
    return filename, PSSM_array
    
    
    
def ID_caller():
    ID = filename
    PSSM_header = []
    for char in range(0,len(ID)):
        if ID[char] == '>':
            header = ID[char:char+7]
            PSSM_header.append(header)
    print(PSSM_header)
    return PSSM_header
    
    
def topo_caller(dataset):
    header_list = []
    s_header_list=[]
    topo_list = []
    match_topo_list =[]
    with open(dataset, 'r') as datahandle:
        all_lines = datahandle.readlines()
    for i in range(0, len(all_lines)):
        if '>' in all_lines[i]:
            header_list.append(all_lines[i].strip())
            topo_list.append(all_lines[i+2].strip())
    for header in header_list:
        x = (header[0:7])
        s_header_list.append(x)
    match = set(PSSM_header).intersection(s_header_list)
    for ID in range(0, len(s_header_list)):
        if s_header_list[ID] in match:
            match_topo_list.append(topo_list[ID])
    return match_topo_list
    
    
if __name__ == '__main__':
    filename, PSSM_array = PSSM_ASCII_open('../Datasets/PSSM/>d1a04a2.c.23.1.1_FASTA.txt.pssm')
    PSSM_header = ID_caller()
    match_topo_list topo_caller('../Datasets/testfilesize50.txt')