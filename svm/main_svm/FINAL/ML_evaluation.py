import matplotlib.pyplot as plt
import pandas as pd #trengs denne?
import config # trengs denne?
import seaborn as sns
import numpy as np #trengs denne?
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.inspection import permutation_importance

def confusion_matrix_heatmap(file_name, target, predicted, model, perc=False):
    """
    Prints a heatmap for better interpration of the results
    
        param target: the labels for the test set, list
        param predicted: the predicted values for the test set, list
        
        return:
    """
    plt.figure()
    plt.title("Model: SVM")# + str(model).split('(')[0] + " Patient: " + str(config.start_patient) + " to " + str(config.end_patient))
    data = {'y_Actual': target, 'y_Predicted': predicted}
    dataset = pd.DataFrame(data, columns=['y_Predicted','y_Actual'])
    c_matrix_l = pd.crosstab(dataset['y_Predicted'], dataset['y_Actual'],
        rownames=['Predicted'], colnames=['Actual'])
    if perc:
        sns.heatmap(c_matrix_l/np.sum(c_matrix_l),
            annot=True, fmt='.2%', cmap='Blues')
    else:
        sns.heatmap(c_matrix_l, annot=True, fmt='d', cmap='Blues')
    tp = c_matrix_l.iloc[0][0]
    tn = c_matrix_l.iloc[1][1] 
    fp = c_matrix_l.iloc[0][1] 
    fn = c_matrix_l.iloc[1][0] 
    sensitivity = tp/(tp+fn) 
    specificity = tn/(fp+tn)
    print("sensitivity", round(sensitivity,3), "specificity", round(specificity,3))

    plt.savefig('heat_' + file_name)
    plt.show()
    plt.close()

def get_metrices(y_test, y_train, y_model_predict_test, _average="macro"): #, cmatrix_plot=False, _print=False, _title="", _average="micro", "macro"
    """
    Prints different values to measure the preformance of the algorithm
    
        param y_test: the labels for the test set, list
        param y_train: the labels for the training set, list
        param y_model_predict_test: the predicted values for the test set, list
        
        return:        
    """    
    _acc_test = round(accuracy_score(y_test, y_model_predict_test), 3)
    _fscore = round(f1_score(y_test, y_model_predict_test, average=_average), 3)
    _precision = round(precision_score(y_test, y_model_predict_test, average=_average), 3)
    _recall = round(recall_score(y_test, y_model_predict_test, average=_average), 3)
    
    print( "Accuracy test: ",_acc_test, "f1 score: ", _fscore, "Precision: ", _precision, "Recall: ", _recall)
    return _acc_test
    
def cross_validation(model,X,y, k):
    """
    Prints the mean AUC score from cross validation
    
        param model: machine learning model after fitting
        param X: features from the data, matrix with features as rows and samples as columns, list of lists
        param y: labels for the data, list of 0 (no seizure) or 1 (seizure)
        
        return:         
    """
    cv_score = cross_val_score(model, X, y, cv=k, scoring='roc_auc')
    print("Mean AUC Score from cross validation: ", cv_score.mean())


def calculate_auc(predicted, y_test):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted)
    #TPR (True Positive Rate or Recall) and FPR (False Positive Rate) 
    # where the former is on y-axis and the latter is on x-axis.
    print('AUC:', round(metrics.auc(fpr, tpr), 3))
    
"""
# denne funker ikke fordi den tar alle channels (32) og sammenlikner md 20 
# vet ikke om den egt er relevant å ha med uansett
def plot_channel_importances_permutations(classifier, X_test, y_test, channel_names):
    result = permutation_importance(classifier, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
    classifier_importances = pd.Series(result.importances_mean, index=channel_names)

    fig, ax = plt.subplots()
    classifier_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()
"""

"""
Old setup:
def plot_importances(file_name, model, names, X_test, y_test, find_important_channels=False, find_important_features=False, mdi = False, mda = False):
    if str(model).split('(')[0] == "SVC":
        importances = model.coef_.ravel()
    elif str(model).split('(')[0] == "RandomForestClassifier" or str(model).split('(')[0] == "GradientBoostingClassifier":
        if mdi: 
            importances = model.feature_importances_
        else: #mda
            result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
            forest_importances = pd.Series(result.importances_mean, index=names)
    else: 
        print("No model named ", model)
        
    importance_unsorted = {}
    for feature_no, value in enumerate(importances):
        importance_unsorted[feature_no] = value
        
    if find_important_features:
        importance_sorted = {}
        for feature_no in range(len(names)): 
            importance_sorted[feature_no] = 0
            for i in range(feature_no,len(importance_unsorted.keys()), len(names)):
                importance_sorted[feature_no] += importance_unsorted[int(i)]  
    elif find_important_channels: 
        importance_sorted = importance_unsorted # bc the channels are already sorted
        
    else: 
        print("Find important channels/features not choosen")
        
    fig = plt.figure()
    fig.set_figheight(8) # make the plot long to show labels on x axis
    if mdi: 
        plt.bar(range(len(importance_sorted)), list(importance_sorted.values()))
    else: # mda
        forest_importances.plot.bar(yerr=result.importances_std)#, ax=ax)
    plt.xticks(range(len(importance_sorted)), names, rotation='vertical')
    plt.yticks("Mean accuracy decrease")
    plt.xlabel("Mean decrease in impurity")
    
    plt.savefig(file_name)
    #plt.show()
    plt.close()

    return importance_sorted
"""
    

def plot_importances(file_name, model, names, X_test, y_test, find_important_channels=False, find_important_features=False, mdi=False, mda=False):
    if str(model).split('(')[0] == "SVC":
        importances = model.coef_.ravel()
    elif str(model).split('(')[0] == "RandomForestClassifier" or str(model).split('(')[0] == "GradientBoostingClassifier":
        if mdi: 
            importances = model.feature_importances_
        elif mda:
            importances = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2).importances_mean
    else: 
        print("No model named ", model)
    
    # HVORFOR HETER DE SORTED OG UNSOTED?
    importance_unsorted = {}
    for feature_no, value in enumerate(importances): 
        importance_unsorted[feature_no] = value # 0,1,2,3... med tilhørende values
        
    if find_important_features:
        importance_sorted = {}
        for feature_no in range(len(names)): # feature names
            importance_sorted[feature_no] = 0
            for i in range(feature_no,len(importance_unsorted.keys()), len(names)):
                importance_sorted[feature_no] += importance_unsorted[int(i)]  
    elif find_important_channels: 
        importance_sorted = importance_unsorted # bc the channels are already sorted and features from seizure and seizure free periods are added in one long column
    else: 
        print("Find important channels/features not choosen")
        
    fig = plt.figure()
    fig.set_figheight(6) # make the plot long to show labels on x axis
    #if mdi: 
    plt.bar(range(len(importance_sorted)), list(importance_sorted.values()))
    #else: # mda
    #    plt.bar(importances, yerr=result.importances_std)#, ax=ax)
    plt.xticks(range(len(importance_sorted)), names, rotation='vertical')
    if mdi:
        plt.ylabel("Weights")#plt.ylabel("Mean decrease in impurity")
    elif mda:
        plt.ylabel("Weights")#plt.ylabel("Mean accuracy decrease")
    else: # svm
        plt.ylabel("Weights")
    if find_important_features:
        plt.xlabel("Feature")
    elif find_important_channels:
        plt.xlabel("Channel")

    plt.savefig(file_name)
    
    
    
    plt.show()
    plt.close()

    return importance_sorted


def evaluate_model(file_name, model, predicted, column_names, X_test, y_test, y_train, features, labels, find_important_channels, find_important_features, mdi, mda):
    if find_important_channels:
        print("\n Results from ",  str(model).split('(')[0], "using channel importance")
    elif find_important_features:
        print("\n Results from ",  str(model).split('(')[0], "using feature importance")
    accuracy = get_metrices(y_test, y_train,predicted)
    confusion_matrix_heatmap(file_name, y_test, predicted, str(model))
    calculate_auc(predicted, y_test)
    #cross_validation(model, features,labels, config.k_folds)
    if find_important_channels: # TODO: this is done in the main function too
        names = []
        for name in column_names[:len(column_names)]:
            if name[:-2] not in names: # the channel names are duplicates (with a trailing 0 or 1)
                names.append(name[:-2])
    elif find_important_features: 
        names = column_names
    else: 
        print("Channel/feature selection not choosen")
    importance_sorted = plot_importances(file_name, model, names, X_test, y_test, find_important_channels, find_important_features, mdi, mda)
    return importance_sorted, accuracy


def plot_accuracies(accuracies, importances_decending):
    fig = plt.figure()
    plt.plot(accuracies, '-ok')
    plt.xlabel("Number of channels")
    plt.xticks(np.arange(len(importances_decending.keys())), np.arange(1, len(importances_decending.keys())+1), rotation =60) 
    plt.ylabel("Accuracy")
    plt.ylim([0,1])
    plt.tight_layout()
    #plt.savefig("accuracies_01.eps")
    plt.show()
    plt.close()
    
    plt.plot(accuracies, '-ok')
    plt.xlabel("Number of channels")
    plt.xticks(np.arange(len(importances_decending.keys())), np.arange(1, len(importances_decending.keys())+1), rotation =60) 
    plt.ylabel("Accuracy")
    plt.tight_layout()
    #plt.savefig("accuracies_zoom.eps")
    plt.show()
    plt.close()
    