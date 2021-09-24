import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.linear_model import OrthogonalMatchingPursuit

#Lasso中的正交匹配追踪
def omp_omp(data,label,n_nonzero_coefs=100):
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
    omp.fit(data, label)
    coef = omp.coef_
    idx_r, = coef.nonzero()
    new_data=data[:,idx_r]
    return new_data,idx_r

data_input = pd.read_excel(r'RPI488_GTPC_1736D.xlsx')
data_ = np.array(data_input)
data = data_[:,1:]
label = data_[:,0]
Zongshu = scale(data)
RNA_shu = Zongshu[:,0:836]
pro_shu = Zongshu[:,836:]

new_RNA_data,index_RNA = omp_omp(RNA_shu,label,n_nonzero_coefs=26)
new_pro_data,index_pro = omp_omp(pro_shu,label,n_nonzero_coefs=66)

data_new = np.hstack((new_RNA_data,new_pro_data))
optimal_RPI_features = pd.DataFrame(data=data_new)




