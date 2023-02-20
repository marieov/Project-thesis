import numpy as np 
import json
import config
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.inspection import permutation_importance


def get_metrices(y_test, y_model_predict_test, _average="macro"): #, cmatrix_plot=False, _print=False, _title="", _average="micro", "macro"
    """
    Prints different values to measure the preformance of the algorithm
    
        param y_test: the labels for the test set, list
        param y_model_predict_test: the predicted values for the test set, list
        
        return:        
    """    
    _acc_test = round(accuracy_score(y_test, y_model_predict_test), 3)
    _fscore = round(f1_score(y_test, y_model_predict_test, average=_average), 3)
    _precision = round(precision_score(y_test, y_model_predict_test, average=_average), 3) 
    _recall = round(recall_score(y_test, y_model_predict_test, average=_average), 3)
    
    print( "Accuracy test: ",_acc_test, "f1 score: ", _fscore)#"Precision: ", _precision, "Recall: ", _recall)
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
    print('AUC:', round(metrics.auc(fpr, tpr), 3))
    
    
def calculate_importances(fitted_model, X_test, y_test, mdi=False, mda=False):
    if config.MODEL == "svm": #str(model).split('(')[0] == "SVC":
        importances = fitted_model.coef_.ravel()
    # rf or gb:
    if mdi: 
        importances = fitted_model.feature_importances_
    elif mda:
        importances = permutation_importance(fitted_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2).importances_mean
    else: 
        print("mda/mdi not choosen")
    return importances


def sort_importances(importances, find_important_features, find_important_channels, names):
    """
    Sort so that importances for the same features/channels are added
    """
    importance_unsorted = {}
    for feature_no, value in enumerate(importances): 
        importance_unsorted[feature_no] = value # 0,1,2,3... med tilhørende values
        
    if find_important_features:
        importance_sorted = {}
        for feature_no in range(len(names)): # feature names
            importance_sorted[feature_no] = 0
            for i in range(feature_no,len(importance_unsorted.keys()), len(names)):
                importance_sorted[feature_no] += importance_unsorted[int(i)]  
        # saved to be used for plotting later 
        np.savetxt("feature_names.txt", names,  fmt="%s")
        json.dump(importance_sorted, open('importance_features_sorted.txt','w'))
        #np.savetxt('importance_features_sorted.txt',  np.array([importance_sorted]), fmt="%s") #_mda' + str(mda) + 'mdi' + str(mdi) + 'tree' + str(tree) + 'feature_channel' + str(find_important_features) + str(find_important_channels) + '.txt', np.array([importance_sorted]), fmt="%s")
    
    elif find_important_channels: 
        importance_sorted = importance_unsorted # bc the channels are already sorted and features from seizure and seizure free periods are added in one long column
        # saved to be used for plotting later 
        np.savetxt("channel_names.txt", names,  fmt="%s")
        json.dump(importance_sorted, open('importance_channels_sorted.txt','w'))
        #np.savetxt('importance_channels_sorted.txt',  np.array([importance_sorted]), fmt="%s") #_mda' + str(mda) + 'mdi' + str(mdi) + 'tree' + str(tree) + 'feature_channel' + str(find_important_features) + str(find_important_channels) + '.txt', np.array([importance_sorted]), fmt="%s")
    
    else: 
        print("Find important channels/features not choosen")

    return importance_sorted


def evaluate_model(fitted_model, predicted, column_names, X_test, y_test, find_important_channels, find_important_features, mdi, mda):
    if find_important_channels:
        print("\n Results from ", config.MODEL, "using channel importance")
    elif find_important_features:
        print("\n Results from ",  config.MODEL, "using feature importance")

    # Names:
    if find_important_channels:
        names = []
        for name in column_names[:len(column_names)]:
            if name[:-2] not in names: 
                names.append(name[:-2]) # the channel names without duplicates (with a trailing 0 or 1)
    elif find_important_features: 
        names = column_names
    else: 
        print("Channel/feature selection not choosen")

    calculate_auc(predicted, y_test)
    #cross_validation(model, features,labels, config.K_FOLDS) # TODO: not working, will fix this later
    importances = calculate_importances(fitted_model, X_test, y_test, mdi, mda)
    sort_importances(importances, find_important_features, find_important_channels, names) # saved to file

    if find_important_features:
        accuracy = get_metrices(y_test,predicted)
        return accuracy
