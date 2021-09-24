import sys,re,os
import pandas as pd
import numpy as np
import itertools
import platform
from math import sqrt

ALPHABET='ACGU'

def readRNAFasta(file):
	with open(file) as f:
		records = f.read()

	if re.search('>', records) == None:
		print('The input RNA sequence must be fasta format.')
		sys.exit(1)
	records = records.split('>')[1:]
	myFasta = []
	for fasta in records:
		array = fasta.split('\n')
		name, sequence = array[0].split()[0], re.sub('[^ACGU-]', '-', ''.join(array[1:]).upper())
		myFasta.append([name, sequence])
	return myFasta
#
def frequency(t1_str, t2_str):

    i, j, tar_count = 0, 0, 0
    len_tol_str = len(t1_str)
    len_tar_str = len(t2_str)
    while i < len_tol_str and j < len_tar_str:
        if t1_str[i] == t2_str[j]:
            i += 1
            j += 1
            if j >= len_tar_str:
                tar_count += 1
                i = i - j + 1
                j = 0
        else:
            i = i - j + 1
            j = 0
    return tar_count

def generate_list(k, alphabet):
    ACGU_list=["".join(e) for e in itertools.product(alphabet, repeat=k)]
    return ACGU_list
#        
def convert_dict(property_index):

    len_index_value = len(property_index[0])
    k = 0
    for i in range(1, 10):
        if len_index_value < 4**i:
            error_infor = 'error, the number of each index value is must be 4^k.'
            sys.stdout.write(error_infor)
            sys.exit(0)
        if len_index_value == 4**i:
            k = i
            break
    kmer_list = generate_list(k, ALPHABET)   
    len_kmer = len(kmer_list)
    phyche_index_dict = {}
    for kmer in kmer_list:
        phyche_index_dict[kmer] = []
    property_index = list(zip(*property_index))
    for i in range(len_kmer):
        phyche_index_dict[kmer_list[i]] = list(property_index[i])
    return phyche_index_dict

def standard(value_list):

    n = len(value_list)
    average_value = sum(value_list) * 1.0 / n
    std_value=sqrt(sum([pow(e - average_value, 2) for e in value_list]) * 1.0 / (n - 1))
    return std_value

def normalize_index(phyche_index, is_convert_dict=False):

    normalize_phyche = []
    for phyche_value in phyche_index:
        average_phyche = sum(phyche_value) * 1.0 / len(phyche_value)
        sd_phyche = standard(phyche_value)
        normalize_phyche.append([round((e - average_phyche) / sd_phyche, 2) for e in phyche_value])
    if is_convert_dict is True:
        return convert_dict(normalize_phyche)
    return normalize_phyche

def parallel_cor_function(nucleotide1, nucleotide2, phyche_index):
    temp_sum = 0.0
    phyche_index_values = list(phyche_index.values())
    len_phyche_index = len(phyche_index_values[0])
    for u in range(len_phyche_index):
        temp_sum += pow(float(phyche_index[nucleotide1][u]) - float(phyche_index[nucleotide2][u]), 2)

    parallel_value=temp_sum / len_phyche_index
    return parallel_value

def get_parallel_factor(k, lamada, sequence, phyche_value):

    theta = []
    l = len(sequence)

    for i in range(1, lamada + 1):
        temp_sum = 0.0
        for j in range(0, l - k - i + 1):
            nucleotide1 = sequence[j: j+k]
            nucleotide2 = sequence[j+i: j+i+k]
            temp_sum += parallel_cor_function(nucleotide1, nucleotide2, phyche_value)
        theta.append(temp_sum / (l - k - i + 1))
    return theta

def make_pseknc_vector(sequence_list, lamada, w, k, phyche_value, theta_type=1):

    kmer = generate_list(k, ALPHABET)
    header = ['#']
    for f in range((16+lamada)):
        header.append('pseknc.'+str(f))
    vector=[]
    vector.append(header)
    for sequence_ in sequence_list:
        name,sequence=sequence_[0],sequence_[1]
        if len(sequence) < k or lamada + k > len(sequence):
            error_info = "Error, the sequence length must be larger than " + str(lamada + k)
            sys.stderr.write(error_info)
        fre_list = [frequency(sequence, str(key)) for key in kmer]
        fre_sum = float(sum(fre_list))
        fre_list = [e / fre_sum for e in fre_list]
        if 1 == theta_type:
            theta_list = get_parallel_factor(k, lamada, sequence, phyche_value)
        theta_sum = sum(theta_list)
        denominator = 1 + w * theta_sum
        
        temp_vec = [round(f / denominator, 3) for f in fre_list]      
        for theta in theta_list:
            temp_vec.append(round(w * theta / denominator, 4))
        sample=[name]
        sample=sample+temp_vec
        vector.append(sample)
    return vector

def kmerArray(sequence, k):
    kmer = []
    for i in range(len(sequence) - k + 1):
        kmer.append(sequence[i:i + k])
    return kmer

phy1=pd.read_csv('phy.csv')
phyche=np.array(phy1)
property_dict=normalize_index(phyche, is_convert_dict=True)
      
def get_property_dict(): 
    file_path =os.path.split(os.path.realpath(__file__))[0] + r'\data\phy.csv' if platform.system() == 'Windows' else  os.path.split(os.path.realpath(__file__))[0] + '/data/phy.csv'
    phy=pd.read_csv(file_path,header=-1,index_col=None)
    phyche=np.array(phy)
    property_dict=normalize_index(phyche, is_convert_dict=True)
    return property_dict

def Psednc(input_data,lamada=10, w=0.05, k = 2):

    phyche_value = property_dict    
    fastas=readRNAFasta(input_data)
    vector = make_pseknc_vector(fastas, lamada, w, k, phyche_value, theta_type=1)
    return vector

vector=Psednc('C_RNA_P_biaohao.txt',lamada=12)
csv_data=pd.DataFrame(data=vector)
