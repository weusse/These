#!/usr/bin/env python
# coding: utf-8


##########################################utils librairies ##########################################
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt 
from matplotlib.ticker import NullFormatter
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import random
random.seed(1000)
from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

##########################################chargement des données ######################################################

def input_data(path):
    palu=pd.read_excel(path)
    return palu


#################################################################################################################
def creat_dataFrame(data):
    paluu=pd.DataFrame(data, columns = ['PPOIDS', 'TEMPERATURE', 'S_M8_APPETIT', 'S_FATIGUE', 'S_ARTHRALGI',
                                    'S_T_DIGESTIF', 'S_VERTIGE', 'S_FRISSON', 'S_MYALGIE', 'S_DABDO', 
                                    'S_VOMISS', 'S_NAUSEE', 'S_CEPHALE', 'S_FIEVRE','TDR'])
    return paluu

###########################Split##########################################################################################
# Nous séparons notre variable cible en deux parties. Une partie pour entrainer y_train et une partie pour tester y_test
def split_data(palu, paluu):
    X = paluu
    y = palu['Diagnostic']
    X1_train, MX_test, y1_train, My_test = train_test_split(X, y, random_state=0)
    return  X1_train, MX_test, y1_train, My_test 

#####################Suréchantillonnage avec l'algorithme SMOTE################################################################
def smote(X1_train,y1_train):
    Re= SMOTE(random_state=0)
    #X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=0)
    columns = X1_train.columns
    Re_data_X1,Re_data_y1=Re.fit_sample(X1_train, y1_train)
    Re_data_X1 = pd.DataFrame(data=Re_data_X1 ,columns= columns)
    X_train =Re_data_X1
    y_train=Re_data_y1
    return  X_train, y_train
##############################################################################################################################
#def log_regression((X_train,y_train,MX_test,My_test):
  #  LR= LogisticRegression(random_state=0,solver='lbfgs',max_iter=1000)
   # LR.fit(X_train, y_train)
   # palu_pred = LR.predict(MX_test)
    #palu_pred=pd.DataFrame(palu_pred)
    #score = metrics.accuracy_score(My_test, y_pred1)
############################################################################################################################
def log_regression(X_train,y_train,MX_test,My_test):
    LR= LogisticRegression(random_state=0,solver='lbfgs',max_iter=1000)
    LR.fit(X_train, y_train)
    palu_pred = LR.predict(MX_test)
    palu_pred=pd.DataFrame(palu_pred)
    score = metrics.accuracy_score(My_test, palu_pred)
    report= classification_report(My_test, palu_pred)
    score=print(score)
    report=print(report)
    return palu_pred, score,report

################################ Decision TreeClassifier #############################################################################
def DecisionTreeClassifier(X_train,y_train,MX_test,My_test):
    from sklearn.tree import DecisionTreeClassifier
    epochs = [1,5,10, 15, 20,25]# List to store the average RMSE for each value of max_depth:
    accuracy = []
    for n in epochs:
        clf = DecisionTreeClassifier(max_depth = n,   random_state = 0)
        clf.fit(X_train, y_train) 
        palu_pred=clf.predict(MX_test)
        score = metrics.accuracy_score(My_test, palu_pred)
        accuracy.append(score)
        ind=accuracy.index(max(accuracy))
        best_score=accuracy[ind]
        report= classification_report(My_test, palu_pred)
    score=print(score)
    report=print(report)
    return palu_pred, score,report
        
        
#################################################### RandomForestClassifier ####################################################
def RandomForestClassifier(X_train,y_train,MX_test,My_test):
    from sklearn.ensemble import RandomForestClassifier
    epochs = [10,50,100, 200, 300,400, 500]# List to store the average RMSE for each value of max_depth:
    accuracy = []
    for n in epochs:
        rf=RandomForestClassifier(n_estimators= n)
        #Train the model using the training sets y_pred=clf.predict(X_test)
        rf.fit(X_train,y_train)
        y_pred=rf.predict(MX_test)
        score = metrics.accuracy_score(My_test, y_pred)
        accuracy.append(score)
        ind=accuracy.index(max(accuracy))
        best_score=accuracy[ind]
        report=classification_report(My_test, y_pred)
    score=print(score)
    report=print(report)
    return y_pred, score,report
############################################ SVM Classifier kernel= linear ################################################################
def linear_svm(X_train,y_train,MX_test,My_test):
    svclassifier1 = SVC(kernel='linear', gamma='auto',probability=True)  
    svclassifier1.fit(X_train, y_train)  
    y_pred1 = svclassifier1.predict(MX_test)    
    score=metrics.accuracy_score(My_test, y_pred1)
    #matrix=confusion_matrix(My_test, y_pred1)
    report=classification_report(My_test, y_pred1)
    score=print(score)
    report=print(report)
    return  y_pred1, score, report
###########################################SVM Classifier kernel= sigmoid ################################################################

def sigmoid_svm(X_train,y_train,MX_test,My_test):
    svclassifier2 = SVC(kernel='sigmoid', gamma='auto',probability=True)  
    svclassifier2.fit(X_train, y_train)  
    y_pred2 = svclassifier2.predict(MX_test)  
    score=metrics.accuracy_score(My_test, y_pred2)
    #matrix=confusion_matrix(My_test, y_pred1)
    rport=classification_report(My_test, y_pred2)
    return y_pred2, score, rport

###################################################SVM Classifier kernel= gaussien ####################################################

def gaussien_svm(X_train,y_train,MX_test,My_test):
    svclassifier = SVC(kernel='rbf', gamma='auto',probability=True)  
    svclassifier.fit(X_train, y_train) 
    y_pred = svclassifier.predict(MX_test) 
    score=metrics.accuracy_score(My_test, y_pred)
    #matrix=confusion_matrix(My_test, y_pred1)
    report=classification_report(My_test, y_pred)
    return y_pred,score,report

##############################################  MLPClassifier #################################################################
def MLPClassifier(X_train,y_train,MX_test,My_test):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    epochs = [1,5,10, 15, 20,25]# List to store the average RMSE for each value of max_depth:
    scaler = StandardScaler()
    scaler.fit(X_train)
    X1_train = scaler.transform(X_train)
    X1_test = scaler.transform(MX_test)
    accuracy = []
    for n in epochs:
        #clf = DecisionTreeClassifier(max_depth = depth,   random_state = 0)
        mlp = MLPClassifier(hidden_layer_sizes=(15,15,n),max_iter=200, solver='lbfgs')
        #clf.fit(X_train, y_train) 
        mlp.fit(X1_train,y_train)
        #y_pred1=clf.predict(MX_test)
        predictions = mlp.predict(X1_test)
        score = metrics.accuracy_score(My_test, predictions)
        accuracy.append(score)
        ind=accuracy.index(max(accuracy))
        best_score=accuracy[ind]
        reported=classification_report(My_test, predictions)
    score=print(best_score)
    reported=print(reported)
    return predictions,best_score,reported

#########################################################################################################################

def plot_data(data,predicted):
# plot with various axes scales
    plt.figure(figsize=[8,4],dpi=200)
    logit_roc_auc = roc_auc_score(data, predicted)
    fpr, tpr, thresholds = roc_curve(data,predicted)
    plt.plot(fpr, tpr, label='logistic regression(area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Courbe ROC')
    plt.legend(loc="lower right")
    #plt.savefig('Log_ROC')
    plt.show()
######################################################################################################################################




















