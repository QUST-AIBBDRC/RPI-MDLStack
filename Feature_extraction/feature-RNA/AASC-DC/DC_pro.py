import sys, os
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
import readFasta
import checkFasta
import numpy as np
import pandas as pd


def DC_pro(fastas, gap=5, **kw):
	if gap < 0:
		print('Error: the gap should be equal or greater than zero' + '\n\n')
		return 0

	if checkFasta.minSequenceLength(fastas) < gap+2:
		print('Error: all the sequence length should be larger than the (gap value) + 2 = ' + str(gap+2) + '\n\n')
		return 0

	AA = 'ACDEFGHIKLMNPQRSTVWYO'
#   AA = 'ACDEFGHIKLMNPQRSTVWYO'
	encodings = []
	aaPairs = []
	for aa1 in AA:
		for aa2 in AA:
			aaPairs.append(aa1 + aa2)
	header = ['#']
	for g in range(gap+1):
		for aa in aaPairs:
			header.append(aa + '.gap' + str(g))
	encodings.append(header)
	for i in fastas:
		name, sequence = i[0], i[1]
		code = [name]
		for g in range(gap+1):
			myDict = {}
			for pair in aaPairs:
				myDict[pair] = 0
			sum = 0
			for index1 in range(len(sequence)):
				index2 = index1 + gap + 1
				if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[index2] in AA:
					myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
					sum = sum + 1
			for pair in aaPairs:
				code.append(myDict[pair] / sum)
		encodings.append(code)
	return encodings

    
kw=  {'path': r"2241RNA_T_proRPI2241_838RNA_Tprotein",}   
fastas1 = readFasta.readFasta(r"RPI2241_838RNA_Tprotein.txt")
result=DC_pro(fastas1, gap=0, **kw)
data1=np.matrix(result[1:])[:,1:]
data_=pd.DataFrame(data=data1)
data_final=pd.DataFrame(data1[:,0:440])
# data_final=pd.DataFrame(data1[:,0:440])
data_final.to_csv('RPI2241_838RNA_Tprotein_TDC.csv')

# kw=  {'path': r"RPI7317_protein_N6.",}   
# fastas1 = readFasta.readFasta(r"RPI7317_protein_N6.txt")
# result=DC_pro(fastas1, gap=0, **kw)
# data1=np.matrix(result[1:])[:,1:]
# data_=pd.DataFrame(data=data1)
# data_final=pd.DataFrame(data1[:,0:400])
# # data_final=pd.DataFrame(data1[:,0:440])
# data_final.to_csv('RPI7317_protein_N6_DC.csv')

# kw=  {'path': r"RPI7317_protein_N10.",}   
# fastas1 = readFasta.readFasta(r"RPI7317_protein_N10.txt")
# result=DC_pro(fastas1, gap=0, **kw)
# data1=np.matrix(result[1:])[:,1:]
# data_=pd.DataFrame(data=data1)
# data_final=pd.DataFrame(data1[:,0:400])
# # data_final=pd.DataFrame(data1[:,0:440])
# data_final.to_csv('RPI7317_protein_N10_DC.csv')

#kw=  {'path': r"RPI1446_protein_P_biaohao.",}   
#fastas1 = readFasta.readFasta(r"RPI1446_protein_P_biaohao.txt")
#result=DC_pro(fastas1, gap=0, **kw)
#data1=np.matrix(result[1:])[:,1:]
#data_=pd.DataFrame(data=data1)
#data_final=pd.DataFrame(data1[:,0:400])
#data_final.to_csv('DC_RPI1446_pro_P.csv')
















