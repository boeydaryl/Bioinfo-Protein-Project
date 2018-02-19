#3 Line FASTA Parser
import pandas as pd

def threelineparser(filename):
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
    print(datadict)
    df = pd.DataFrame.from_dict(data=datadict, orient='index')
    print(df)




if __name__ == "__main__":
    threelineparser('/Users/daryl/Documents/Bioinfo-Protein-Project/Project/Datasets/testfile.txt')