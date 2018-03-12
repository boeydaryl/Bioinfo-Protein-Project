#PSSM Bash script

cd ../Datasets/FASTA_new/

for element in *.txt

do

psiblast -query $element -out_ascii_pssm ../PSSM/$element.pssm -db ~/Documents/blastdb/uniprot_sprot.fasta -num_iterations 3 -evalue 0.01

done