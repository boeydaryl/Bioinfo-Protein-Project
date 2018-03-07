def SVMTest(model, testfilename):
    clf = joblib.load(model)
    file1 = open(testfilename, 'r')
    print(file1)
    for x in file1:
        print(x)
        
        
        
if __name__ == "__main__":
    SVMTest('output.pkl', '/Users/daryl/Documents/Bioinfo-Protein-Project/Project/Datasets/5W0P_A.fasta.txt')