# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 11:17:36 2021

@author: xw
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.manifold import SpectralEmbedding 

def SE_select(data,n_components=300):
    embedding = SpectralEmbedding(n_components=n_components)
    X_transformed = embedding.fit_transform(data)
    return X_transformed

data_input = pd.read_excel(r'RPI488_GTPC_1736D.xlsx')
data_ = np.array(data_input)
data = data_[:,1:]
label = data_[:,0]
Zongshu = scale(data)
RNA_shu = Zongshu[:,0:836]
pro_shu = Zongshu[:,836:]

new_RNA_data = SE_select(RNA_shu, n_components=26)
new_pro_data = SE_select(pro_shu, n_components=66)

data_new = np.hstack((new_RNA_data,new_pro_data))
optimal_RPI_features = pd.DataFrame(data=data_new)
