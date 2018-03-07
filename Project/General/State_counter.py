#State counter

statelist = []
file1 = open('/Users/daryl/Documents/Bioinfo-Protein-Project/Project/Datasets/testpart1.txt', 'r')
file2 = open('/Users/daryl/Documents/Bioinfo-Protein-Project/Project/Datasets/testpart2.txt', 'r')
file3 = open('/Users/daryl/Documents/Bioinfo-Protein-Project/Project/Datasets/testpart3.txt', 'r')
for x in file3:
    if x.islower() and '>' not in x:
        statelist.extend(x.replace("\n", ''))
#print(statelist)
e_count = 0
b_count = 0
for x in statelist:
    if x == 'e':
        e_count += 1
    elif x == 'b':
        b_count += 1
print('E/B ratio =' + " " + str(e_count/b_count))
    