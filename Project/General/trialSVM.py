#Trial SVM classifier

import pandas as pd
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder


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
    

def windowmaker(D):
    
    
    


def classifier(D):
    v = DictVectorizer(sparse=False)
    le = LabelEncoder
    AA = []
    Features = []
    print(D)
    for x, y in D.items():
        for z in y:
            if z.isupper():
                AA
    
    





if __name__ == "__main__":
    threelineparser('/Users/daryl/Documents/Bioinfo-Protein-Project/Project/Datasets/testfile.txt', 'fulloutput.csv')
    windowmaker(datadict)
    classifier(datadict)