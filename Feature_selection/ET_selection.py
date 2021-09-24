import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
#基于L1正则化的决策树选择算法
def selectFromExtraTrees(data,label):
    clf = ExtraTreesClassifier(n_estimators=500, criterion='gini', max_depth=None, 
                               min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                               max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, 
                               min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=1, 
                               random_state=None, verbose=0, warm_start=False, class_weight=None)#entropy
    clf.fit(data,label)
    importance=clf.feature_importances_ 
    model=SelectFromModel(clf,prefit=True)
    new_data = model.transform(data)
    return new_data,importance

data_input = pd.read_excel(r'RPI488_GTPC_1736D.xlsx')
data_ = np.array(data_input)
data = data_[:,1:]
label = data_[:,0]
Zongshu = scale(data)
RNA_shu = Zongshu[:,0:836]
pro_shu = Zongshu[:,836:]

new_RNA_data, index_RNA = selectFromExtraTrees(RNA_shu,label)
feature_numbe_RNA = -index_RNA
H_RNA = np.argsort(feature_numbe_RNA)
mask_RNA = H_RNA[:26]
new_data_RNA = RNA_shu[:,mask_RNA]

new_pro_data, index_pro = selectFromExtraTrees(pro_shu,label)
feature_numbe_pro = -index_pro
H_pro = np.argsort(feature_numbe_pro)
mask_pro = H_pro[:66]
new_data_pro = pro_shu[:,mask_pro]

optimal_RPI_features = np.hstack((new_data_RNA,new_data_pro))





