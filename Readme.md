##RPI-MDLStack

RPI-MDLStack: predicting RNA-protein interactions through deep learning with stacking strategy and LASSO

###PRPI-MDLStack uses the following dependencies:

 * Python 3.6
 * numpy
 * scipy
 * scikit-learn
 * pandas
 * tensorflow 
 * keras

###Guiding principles: 

**The dataset file contains six datasets, among which RPI488, RPI369, RPI2241, RPI1807, RPI1446, NPInter v3.0.

**Feature extraction：
 * feature-RNA is the implementation of kmer, AASC-DC, PseDNC and PseSSC for RNA.
 * feature-protein is the implementation of CT, GTPC, PseAAC and RPT for protein.

**Feature_selection:
 * EN_selection is the implementation of elastic net.
 * ET_selection is the implementation of extra-trees.
 * LASSO_selection is the implementation of LASSO.
 * LLE_selection is the implementation of locally linear embedding.
 * LR_selection is the implementation of logistic regression.
 * MDS_selection is the implementation of multidimensional scaling analysis.
 * MI_selection is the implementation of mutual information.
 * OMP_selection is the implementation of orthogonal matching pursuit.
 * SE_selection is the implementation of spectral embedding.
 * TSVD_selection is the implementation of truncated singular value decomposition.
 
**Classifier:
 * RPI-MDLStack_model.py is the implementation of our model in this work.
 * DNN_singleclassifier.py is the implementation of deep neural network as a single classifier.
 * GRU_singleclassifier.py is the implementation of gated recurrent unit as a single classifier.
 * CNN_singleclassifier.py is the implementation of convolutional neural network as a single classifier.
 * RNN_singleclassifier.py is the implementation of recurrent neural network as a single classifier.
 * MLP_singleclassifier.py is the implementation of multilayer perceptron as a single classifier.
 * SVM_singleclassifier.py is the implementation of support vector machine as a single classifier.
 * RF_singleclassifier.py is the implementation of random forest as a single classifier.
 * AdaBoost_singleclassifier.py is the implementation of adaptive boosting as a single classifier.
 * ET_singleclassifier.py is the implementation of extra-trees as a single classifier.
 * GBDT_singleclassifier.py is the implementation of gradient boosting decision tree as a single classifier.
 * KNN_singleclassifier.py is the implementation of K nearest neighbor as a single classifier.
 * NB_singleclassifier.py is the implementation of Naïve Bayes as a single classifier.


