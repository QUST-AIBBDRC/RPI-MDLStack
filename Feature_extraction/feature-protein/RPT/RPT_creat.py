import scipy.io as sio
import numpy as np 
import pickle as p 
#from sklearn.preprocessing import scale,StandardScaler,MinMaxScaler 
#import matplotlib.pyplot as plt

np.random.seed(100)
first = sio.loadmat('tpc1_python_PSSM.mat')
yeast_PA=first.get('pssm')
index_PA=first.get('index_PA')
CNN_pre_A=[]
num=len(index_PA)

data_list1=[]
data_H=[]
for i in range(num-1):
    H1=index_PA[i].tolist()[0]###tolist将数组或者矩阵转换成列表
    H2=index_PA[i+1].tolist()[0]
    data_H=yeast_PA[H1:H2,:]####data_H表示每条序列对应的pssm矩阵
    data_list1.append(data_H)
    H1=[]
    H2=[]
    data_H=[]


f1 = open('RPI_pssm_PA1.data', 'wb') ###建立一个空的.data文件
p.dump(data_list1, f1) ##把data_list1放进上边建立的空.data文件
f1.close() 
 
import scipy.io as sio
import pickle as p
import numpy as np
LG =  5

def trigrams(matrix):
    return_matrix = []
    for x in range(len(matrix[0])):
      for y in range(len(matrix[0])):
        for z in range(len(matrix[0])):
          for i in range( len(matrix) - 2):
            value = matrix[i][x] * matrix[i+1][y] * matrix[i+2][z]
            return_matrix.append( value )
    return return_matrix

def residue_probing_transform(matrix):####RPT
  features = []
  matrix_sums = list(map(sum, zip(*matrix)))

  for i in range(len(matrix[0])):
    for j in range(len(matrix[0])):
      features.append( ( matrix_sums[i]+ matrix_sums[j]) / len(matrix) )
  return features

f1=open(r'RPI369_N_pssm_PA1.data','rb')
pssm1=p.load(f1)
aac=[]
RPT=[] 
#for i in range(len(pssm1)):
#    aac_pssm_obtain=aac_pssm(pssm1[i])
#    aac.append(aac_pssm_obtain)

for i in range(len(pssm1)):
    RPT_obtain=residue_probing_transform(pssm1[i])
    RPT_obtain1=np.array(RPT_obtain)
    RPT_obtain2=RPT_obtain1.T
    RPT_obtain=np.reshape(RPT_obtain2,(1,400)) 
    RPT.append(RPT_obtain)
RPT1=np.array(RPT)
[m,n,q]=np.shape(RPT1)
RPI_RPT=np.reshape(RPT1,(m,400))















