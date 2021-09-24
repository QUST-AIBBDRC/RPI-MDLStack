import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.linear_model import ElasticNet,ElasticNetCV
#弹性网降维
def EN_select(data,label,alpha =np.array([0.01,0.02,0.03,0.04, 0.05,0.06,0.07,0.08,0.09,0.1])):
    enetCV = ElasticNetCV(alphas=alpha,l1_ratio=0.1,max_iter=2000).fit(data,label)
    enet=ElasticNet(alpha=enetCV.alpha_, l1_ratio=0.1,max_iter=2000)
    enet.fit(data,label)
    mask = enet.coef_ != 0
    new_data = data[:,mask]
    return new_data,mask

data_input = pd.read_excel(r'RPI488_GTPC_1736D.xlsx')
data_ = np.array(data_input)
data = data_[:,1:]
label = data_[:,0]
Zongshu = scale(data)
RNA_shu = Zongshu[:,0:836]
pro_shu = Zongshu[:,836:]
new_RNA_data,index_RNA = EN_select(RNA_shu, label, alpha=np.array([1.85]))
new_pro_data,index_pro = EN_select(pro_shu, label, alpha=np.array([0.22]))

optimal_RPI_features = np.hstack((new_RNA_data,new_pro_data))
