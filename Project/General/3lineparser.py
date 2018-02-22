#3 Line FASTA Parser
import pandas as pd

def threelineparser(filename, outputfilename):
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
    print(datadict)




if __name__ == "__main__":
    threelineparser('/Users/daryl/Documents/Bioinfo-Protein-Project/Project/Datasets/testfile.txt', 'fulloutput.csv')