#FASTA Parser
#Works to parse header and sequence from dataset to individual FASTA file

def parser(filename):
    count = 0
    with open(filename, 'r') as fh1:
        all_lines = fh1.readlines()
    #print(all_lines)
    header_list = []
    seq_list = []
    for i in range(0, len(all_lines)):
        if '>' in all_lines[i]:
            count += 1
            filename1 = '../Datasets/50New/FASTA/' + str(all_lines[i].strip()) + '_FASTA.txt'
            writefile = open(filename1, 'w')
            writefile.write(all_lines[i])
            writefile.write(all_lines[i+1])

            
            
if __name__ == "__main__":
    parser('../Datasets/50newproteins.txt')