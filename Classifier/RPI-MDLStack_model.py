import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from keras.layers.recurrent import GRU
from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras.layers import Flatten
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import math 
from sklearn.metrics import roc_curve, auc

def categorical_probas_to_classes(p):
    return np.argmax(p, axis=1)

def to_categorical(y, nb_classes=None):
    y = np.array(y, dtype='int')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1
    return Y

def calculate_performace(test_num, pred_y,  labels):
    tp =0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] ==1:
            if labels[index] == pred_y[index]:
                tp = tp +1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn +1
            else:
                fp = fp + 1               
            
    acc = float(tp + tn)/test_num
    precision = float(tp)/(tp+ fp + 1e-06)
    npv = float(tn)/(tn + fn + 1e-06)
    sensitivity = float(tp)/ (tp + fn + 1e-06)
    specificity = float(tn)/(tn + fp + 1e-06)
    mcc = float(tp*tn-fp*fn)/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) + 1e-06)
    f1=float(tp*2)/(tp*2+fp+fn+1e-06)
    return acc, precision,npv, sensitivity, specificity, mcc, f1

def get_GRU_model(input_dim,out_dim): 
    model = Sequential()
    model.add(GRU(256,return_sequences=True,input_shape=(1,input_dim))) 
    model.add(Dropout(0.05))
    model.add(GRU(256,return_sequences=True)) 
    model.add(Dropout(0.05))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu',name="Dense_128"))
    model.add(Dropout(0.05))
    model.add(Dense(2, activation = 'softmax',name="Dense_2"))
    model.compile(loss = 'binary_crossentropy', optimizer = 'Adam', metrics =['accuracy'])#rmsprop
    return model

def get_DNN_model(input_dim,out_dim):
    model = Sequential()
    model.add(Dense(int(128), activation='relu', init='glorot_normal', name='Dense__1',input_shape=(1,input_dim)))
    model.add(Dropout(0.05))
    model.add(Dense(int(128), activation='relu', init='glorot_normal', name='Dense__2'))
    model.add(Dropout(0.05))
    model.add(Flatten())
    model.add(Dense(int(64), activation='relu', init='glorot_normal', name='Dense_3'))
    model.add(Dropout(0.05))
    model.add(Dense(2, activation='softmax', name='output'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics =['accuracy'])#rmsprop
    return model

def get_stacking(clf, x_train, y_train, x_test, num_class,n_folds=5):
    kf = KFold(n_splits=n_folds)
    second_level_train_set=[]
    test_nfolds_set=[]
    for i,(train_index, test_index) in enumerate(kf.split(x_train)):
        x_tra, y_tra = x_train[train_index], y_train[train_index]
        x_tst, y_tst =  x_train[test_index], y_train[test_index]
        
        clf.fit(x_tra, y_tra,epochs=30)
        
        second_level_train_ = clf.predict_proba(x_tst)
        second_level_train_set.append(second_level_train_)
        test_nfolds= clf.predict_proba(x_test)
        test_nfolds_set.append(test_nfolds)   
    train_second=second_level_train_set
    train_second_level=np.concatenate((train_second[0],train_second[1],train_second[2],train_second[3],train_second[4]),axis=0) 
    test_second_level_=np.array(test_nfolds_set)   
    test_second_level=np.mean(test_second_level_,axis = 0)
    return train_second_level,test_second_level

def get_stacking2(clf, x_train, y_train, x_test, num_class,n_folds=5):
    kf = KFold(n_splits=n_folds)
    second_level_train_set=[]
    test_nfolds_set=[]
    for i,(train_index, test_index) in enumerate(kf.split(x_train)):
        x_tra, y_tra = x_train[train_index], y_train[train_index]
        x_tst, y_tst =  x_train[test_index], y_train[test_index]
        clf.fit(x_tra, y_tra)
        second_level_train_ = clf.predict_proba(x_tst)
        second_level_train_set.append(second_level_train_)
        test_nfolds= clf.predict_proba(x_test)
        test_nfolds_set.append(test_nfolds)   
    train_second=second_level_train_set
    train_second_level=np.concatenate((train_second[0],train_second[1],train_second[2],train_second[3],train_second[4]),axis=0) 
    test_second_level_=np.array(test_nfolds_set)   
    test_second_level=np.mean(test_second_level_,axis = 0)
    return train_second_level,test_second_level

def get_first1_level(train_x, train_y, test_x,num_class):
    model1 = get_GRU_model(input_dim,out_dim)
    model2 = get_DNN_model(input_dim,out_dim)
    train_sets = []
    test_sets = []
    for clf in [model1, model2]:
        train_set, test_set = get_stacking(clf, train_x, train_y, test_x,num_class)
        train_sets.append(train_set)
        test_sets.append(test_set)
    meta_train1 = np.concatenate([result_set.reshape(-1,1,num_class) for result_set in train_sets], axis=1)
    meta_test1 = np.concatenate([y_test_set.reshape(-1,1,num_class) for y_test_set in test_sets], axis=1)
    return meta_train1,meta_test1

def get_first2_level(train_x, train_y, test_x,num_class):
    model3 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(1000, 100), random_state=10)
    model4 = SVC(probability=True,kernel='rbf')
    model5 = RandomForestClassifier(n_estimators=1000, criterion='gini')
    train_sets = []
    test_sets = []
    for clf in [model3,model4,model5]:
        train_x=np.reshape(train_x,(-1,input_dim))
        test_x=np.reshape(test_x,(-1,input_dim))
        train_y=np.reshape(y_train2,(-1,1))
        train_set, test_set = get_stacking2(clf, train_x, train_y, test_x,num_class)
        train_sets.append(train_set)
        test_sets.append(test_set)
    meta_train2 = np.concatenate([result_set.reshape(-1,num_class) for result_set in train_sets], axis=1)
    meta_test2 = np.concatenate([y_test_set.reshape(-1,num_class) for y_test_set in test_sets], axis=1)
    return meta_train2,meta_test2

def get_second_level(train_dim,train_label,test_dim,num_class):
    meta_train1,meta_test1=get_first1_level(train_dim,train_label,test_dim,num_class)
    [aa,bb,cc]=np.shape(X_train)
    [aa1,bb1,cc1]=np.shape(X_test)
    meta_train1=np.reshape(meta_train1,(aa,4))
    meta_test1=np.reshape(meta_test1,(aa1,4))
    meta_train2,meta_test2=get_first2_level(train_dim,train_label,test_dim,num_class)
    meta_train_fusion=np.hstack((meta_train1,meta_train2))
    meta_test_fusion=np.hstack((meta_test1,meta_test2))
    
    meta_classifier=SVC(probability=True,kernel='rbf')
    
    hist=meta_classifier.fit(meta_train_fusion,y_train2)
    pre_score=meta_classifier.predict_proba(meta_test_fusion)
    return meta_train_fusion,meta_test_fusion,pre_score
 
################################################################################
################################################################################


data_start = pd.read_csv("RPI_Dataset_features.csv")
label_P = np.ones(int('positive_number'))
label_N = np.zeros(int('negative_number'))
label_start = np.hstack((label_P,label_N))
label=np.array(label_start)
data=np.array(data_start)
shu=scale(data)

X_=scale(shu)
y_= label
X,y=(X_,y_)
[sample_num,input_dim]=np.shape(X)

out_dim=2
num_class=2
n=input_dim

sepscores = []
ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5

skf= StratifiedKFold(n_splits=5)

for train, test in skf.split(X,y): 
    X_train=np.reshape(X[train],(-1,1,input_dim))
    X_test=np.reshape(X[test],(-1,1,input_dim))
    y_train=to_categorical(y[train])#generate the resonable results
    y_train2=y[train]
    y_test=to_categorical(y[test])#generate the test 
    [aa,bb,cc]=np.shape(X_train)
    input_dim2=aa
    
    meta_train2,meta_test2,y_score=get_second_level(X_train,y_train,X_test,num_class)

    yscore=np.vstack((yscore,y_score))
    y_test=to_categorical(y[test]) 
    ytest=np.vstack((ytest,y_test))
    fpr, tpr, _ = roc_curve(y_test[:,0], y_score[:,0])
    roc_auc = auc(fpr, tpr)
    y_class= categorical_probas_to_classes(y_score)
    y_test_tmp=y[test]
    acc, precision,npv, sensitivity, specificity, mcc,f1 = calculate_performace(len(y_class), y_class, y_test_tmp)
    sepscores.append([acc, precision,npv, sensitivity, specificity, mcc,f1,roc_auc])
    print('Resultlist: acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f'
          % (acc, precision,npv, sensitivity, specificity, mcc,f1, roc_auc))
    hist=[]
    cv_clf=[]
scores=np.array(sepscores)  
    
print("acc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[0]*100,np.std(scores, axis=0)[0]*100))
print("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[1]*100,np.std(scores, axis=0)[1]*100))
print("npv=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[2]*100,np.std(scores, axis=0)[2]*100))
print("sensitivity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[3]*100,np.std(scores, axis=0)[3]*100))
print("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[4]*100,np.std(scores, axis=0)[4]*100))
print("mcc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[5]*100,np.std(scores, axis=0)[5]*100))
print("f1=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[6]*100,np.std(scores, axis=0)[6]*100))
print("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[7]*100,np.std(scores, axis=0)[7]*100))

result1=np.mean(scores,axis=0)
H1=result1.tolist()
sepscores.append(H1)
result=sepscores[2]

print('Results: ACC=%f,Pr=%f,npv=%f,Se=%f,Sp=%f,Mcc=%f,F1=%f,ROC_AUC=%f'
      % (result[0],result[1],result[2],result[3],
         result[4],result[5], result[6],result[7]))
