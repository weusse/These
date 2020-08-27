
#from utils import *
from model_palu import *
import warnings
warnings.filterwarnings("ignore")

def main_palu():
    path="C:\\Users\\ASUS\\Documents\\these\\code_these-master\\palu0001.xlsx"
    palu=input_data(path)
    paluu=creat_dataFrame(palu)
    X1_train, MX_test, y1_train, My_test=split_data( palu,paluu)
    X_train, y_train=smote(X1_train,y1_train)
    
   # Logistique Regression classifier
    LR_pred=log_regression(X_train,y_train,MX_test,My_test)
    print(LR_pred[1:])

    #plot_data(My_test,LR_pred[0])
    
    #DecisionTreeClassifier
    DT_pred=DecisionTreeClassifier(X_train,y_train,MX_test,My_test)
    print(DT_pred[1:])

    #plot_data(My_test,DT_pred[0])
    
    
    #RandomForestClassifier
    RF_pred=RandomForestClassifier(X_train,y_train,MX_test,My_test)
    print(RF_pred[1:])

    #plot_data(My_test,RF_pred[0])
    
    # linear_svm
   # lsvm_pred=linear_svm(X_train,y_train,MX_test,My_test)
    #print(lsvm_pred[1:])
   # plot_data(My_test,lsvm_pred[0])
    
    #Sigmoid SVM
    #sig_svm=sigmoid_svm(X_train,y_train,MX_test,My_test)
    #print(sig_svm[1:])
    #plot_data(My_tests,sig_svm[0])
    
    #Gaussian SVM
    #gaus_svm=gaussien_svm(X_train,y_train,MX_test,My_test)
    #print(gaus_svm[1:])
    #plot_data(My_test,gaus_svm[0])
    
    #   MLPClassifier
    #mlp_pred= MLPClassifier(X_train,y_train,MX_test,My_test)
    #print(print(mlp_pred[1:]))
    
if __name__ == '__main__':
    main_palu()