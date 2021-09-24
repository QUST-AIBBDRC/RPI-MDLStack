import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

#互信息进行特征选择
def mutual_mutual(data,label,k=300):
    model_mutual= SelectKBest(mutual_info_classif, k=k)
    new_data=model_mutual.fit_transform(data, label)
    mask = model_mutual._get_support_mask()
    return new_data,mask

data_input = pd.read_excel(r'RPI488_GTPC_1736D.xlsx')
data_ = np.array(data_input)
data = data_[:,1:]
label = data_[:,0]
Zongshu = scale(data)
RNA_shu = Zongshu[:,0:836]
pro_shu = Zongshu[:,836:]

new_RNA_data ,index_RNA = mutual_mutual(RNA_shu,label,k=26)
new_pro_data ,index_pro = mutual_mutual(pro_shu,label,k=66)

data_new = np.hstack((new_RNA_data,new_pro_data))
optimal_RPI_features = pd.DataFrame(data=data_new)

