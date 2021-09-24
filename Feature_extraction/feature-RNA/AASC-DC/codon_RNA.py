import numpy as np
import numpy
import readFasta
import re, sys, os, platform
import pandas as pd

codon_table = {
    'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A', 'CGU': 'R', 'CGC': 'R',
    'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R', 'UCU': 'S', 'UCC': 'S',
    'UCA': 'S', 'UCG': 'S', 'AGU': 'S', 'AGC': 'S', 'AUU': 'I', 'AUC': 'I',
    'AUA': 'I', 'UUA': 'L', 'UUG': 'L', 'CUU': 'L', 'CUC': 'L', 'CUA': 'L',
    'CUG': 'L', 'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G', 'GUU': 'V',
    'GUC': 'V', 'GUA': 'V', 'GUG': 'V', 'ACU': 'T', 'ACC': 'T', 'ACA': 'T',
    'ACG': 'T', 'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P', 'AAU': 'N',
    'AAC': 'N', 'GAU': 'D', 'GAC': 'D', 'UGU': 'C', 'UGC': 'C', 'CAA': 'Q',
    'CAG': 'Q', 'GAA': 'E', 'GAG': 'E', 'CAU': 'H', 'CAC': 'H', 'AAA': 'K',
    'AAG': 'K', 'UUU': 'F', 'UUC': 'F', 'UAU': 'Y', 'UAC': 'Y', 'AUG': 'M',
    'UGG': 'W'
}


##将RNA序列转化为3条氨基酸序列，
def translateTheSeq(seq):
    seq1 = ''
    seq2 = ''
    seq3 = ''
    for frame in range(3):
        prot = ''
        for i in range(frame, len(seq), 3):
            codon = seq[i:i + 3]
            if codon in codon_table:
                prot = prot + codon_table[codon]
            else:
                prot = prot + 'O'
        if frame == 0:
            seq1 = prot
        elif frame == 1:
            seq2 = prot
        elif frame == 2:
            seq3 = prot
    seq_finish = seq1 + seq2 + seq3
    return seq_finish

#
#
fastas = readFasta.readFasta("RNA_sequence.txt")
result1=[]

for i in fastas:   
    name, RNA_sequence = i[0], re.sub('-', '', i[1])
    result=translateTheSeq(RNA_sequence)
    result1=(result1,result)



