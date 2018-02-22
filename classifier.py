import pandas as pd
import numpy as np
import os
import csv
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_score, roc_auc_score, confusion_matrix, log_loss, recall_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt

def warn(*args, **kwargs):
    pass

def classifyData():
    global knn, mlp, abc    
    classifiers = ['Neural Network','AdaBoost','K Nearest Neighbors']

    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data")
    y = np.array(df["whether he/she donated blood in March 2007"])
    X = np.array(df.drop(['Monetary (c.c. blood)','whether he/she donated blood in March 2007'],1))

    for index in range(len(classifiers)):
    
        if(classifiers[index] == 'Neural Network'):
            classifier = mlp
        elif(classifiers[index] == 'AdaBoost'):
            classifier = abc
        else:
            classifier = knn
             
        k = 10
        kf = StratifiedKFold(n_splits = k, shuffle = True, random_state = 97)

        accuracy = 0
        precision = 0
        aucScore = 0
        cm = 0
        logloss = 0
        recall = 0
        y_test_final = np.array([])
        y_predict_final = np.array([])
        y_probability_final = np.array([])
        y_id = np.array([])

        for train_index, test_index in kf.split(X,y):
        
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            scaler = StandardScaler()
            # Fit only to the training data
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            classifier = classifier.fit(X_train,y_train)
            y_predict = classifier.predict(X_test)
            y_probability  = classifier.predict_proba(X_test)
                
            accuracy += classifier.score(X_test, y_test, sample_weight=None)
            precision += precision_score(y_test, y_predict, average='macro')
            recall += recall_score(y_test, y_predict, average='macro')
            aucScore +=  roc_auc_score(y_test, y_predict)
            logloss += log_loss(y_test, y_predict)
            
            y_test_final =np.append(y_test_final,y_test)
            y_predict_final =np.append(y_predict_final,y_predict)
            y_id = np.append(y_id,test_index)
            
            for i in range(len(y_probability)):
                y_probability_final = np.append(y_probability_final, round(y_probability[i][1],1))  
        
        print(classifiers[index],': ')
        print('-------------------------------------------')
        print("Accuracy: ", round(accuracy/k*100,2),'%')
        print("Precision: ", round(precision/k*100,2),'%')
        print("Recall: ", round(recall/k*100),'%')
        print("AUC: ", aucScore/k)
        print("logloss: " , logloss/k)
        print("Confusion matrix: ")
        cnf = confusion_matrix(y_test_final, y_predict_final, labels=None)
        print(cnf)
        print('-------------------------------------------')
        print()

        # Roc curve plot
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test_final, y_predict_final)
        roc_auc = auc(false_positive_rate, true_positive_rate)

        plt.clf()
        plt.title('Receiver Operating Characteristic:'+ classifiers[index])
        plt.plot(false_positive_rate, true_positive_rate, 'b',
        label='AUC = %0.2f'% roc_auc)
        plt.legend(loc='lower right', fontsize = 'large')
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([-0.1,1.2])
        plt.ylim([-0.1,1.2])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

        plt.text(x=0.75,y=0.5,s="Accuracy: {:.2f}%"\
                .format(accuracy/k*100),
                fontsize=12)
        plt.text(x=0.75,y=0.4,s="Precision: {:.2f}%"\
                .format(precision/k*100),
                fontsize=12)
        plt.text(x=0.75,y=0.3,s="Recall:     {:.2f}%"\
                .format(recall/k*100),
                fontsize=12)
        plt.text(x=0.75,y=0.2,s="LogLoss:   {:.4f}"\
                .format(logloss/k),
                fontsize=12)
        # plt.show()

        if not os.path.isdir('.//ROC_Curves'):
            os.mkdir('.//ROC_Curves')

        plt.savefig('.//ROC_Curves//'+classifiers[index]+'.png')

        if not os.path.isdir('.//OutputFiles'):
            os.mkdir('.//OutputFiles')

        path = ".//OutputFiles//"+classifiers[index]+"_OutputFile.csv"

        csvfile = open(path,'w',newline='')
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Id','Made Donation in March 2007'])
        # outputFile.write("Id,Made Donation in March 2007\n")
        
        for i in range(len(y_id)):
            # outputFile.write(str(int(y_id[i]+1))+','+str(y_probability_final[i])+'\n')
            writer.writerow([str(int(y_id[i]+1)),str(y_probability_final[i])])
        csvfile.close()


if __name__=='__main__':

    print("Program is executing. Please wait..")
    warnings.warn = warn
    

    # Classifiers used
    #K nearest neighbors 
    knn = KNeighborsClassifier(n_neighbors = 10, 
                            weights       = 'uniform',  # 'uniform', 'distance', a user-defined function 
                            algorithm     = 'brute', 
                            leaf_size     = 30, 
                            p             = 2, 
                            metric        = 'euclidean', # see DistanceMetric class
                            metric_params = None, 
                            n_jobs        = -1)

    #  ANN
    mlp = MLPClassifier(hidden_layer_sizes=(200,10, 100), 
                        activation='relu', 
                        solver='adam', 
                        alpha=0.0001, 
                        batch_size=25, 
                        learning_rate='constant', 
                        learning_rate_init=0.01, 
                        power_t=0.5, max_iter=100, 
                        shuffle=True, 
                        random_state=25, 
                        tol=0.0001, 
                        verbose=False, 
                        warm_start=False, 
                        momentum=0.9, 
                        nesterovs_momentum=True, 
                        early_stopping=False, 
                        validation_fraction=0.1, 
                        beta_1=0.9, 
                        beta_2=0.999, 
                        epsilon=1e-08)
    
    #  Adaboost
    abc = AdaBoostClassifier(base_estimator=None, 
                                n_estimators=20, 
                                learning_rate = 1, 
                                algorithm='SAMME.R', 
                                random_state=None)

    classifyData()


