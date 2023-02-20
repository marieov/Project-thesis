import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import config
import json


def confusion_matrix_heatmap(file_name, target, predicted):
    """
    Prints a heatmap for better interpration of the results
    
        param target: the labels for the test set, list
        param predicted: the predicted values for the test set, list
        
        return:
    """
    plt.figure()
    plt.title("Model: " + str(config.MODEL).split('(')[0])# + " Patient: " + str(config.START_PATIENT) + " to " + str(config.END_PATIENT))
    data = {'y_Actual': target, 'y_Predicted': predicted}
    dataset = pd.DataFrame(data, columns=['y_Predicted','y_Actual'])
    c_matrix_l = pd.crosstab(dataset['y_Predicted'], dataset['y_Actual'],
        rownames=['Predicted'], colnames=['Actual'])
    tp = c_matrix_l.iloc[0][0]
    tn = c_matrix_l.iloc[1][1] 
    fp = c_matrix_l.iloc[0][1] 
    fn = c_matrix_l.iloc[1][0] 
    sensitivity = tp/(tp+fn) 
    specificity = tn/(fp+tn)
    print("sensitivity", round(sensitivity,3), "specificity", round(specificity,3))

    sns.heatmap(c_matrix_l, annot=True, fmt='d', cmap='Blues')
    
    plt.savefig(file_name + config.MODEL + ".pdf")
    plt.show()
    plt.close()


def plot_accuracies(importances_decending):
    accuracies = np.loadtxt(config.ACCURACY_FILE)
    # Full plot
    plt.plot(accuracies, '-ok')
    plt.xlabel("Number of channels")
    plt.xticks(np.arange(len(importances_decending.keys())), np.arange(1, len(importances_decending.keys())+1), rotation =60) 
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig("accuracies_zoom" + config.MODEL + ".pdf")
    plt.show()
    plt.close()

    # Scaled to 0-1
    plt.plot(accuracies, '-ok')
    plt.xlabel("Number of channels")
    plt.xticks(np.arange(len(importances_decending.keys())), np.arange(1, len(importances_decending.keys())+1), rotation =60) 
    plt.ylabel("Accuracy")
    plt.ylim([0,1])
    plt.tight_layout()
    plt.savefig("accuracies_01" + config.MODEL + ".pdf")
    plt.show()
    plt.close()


def plot_importances(importance_sorted, file_name, names, find_important_channels=False, find_important_features=False, mdi=False, mda=False):
    
    plt.bar(range(len(importance_sorted)), list(importance_sorted.values()))
    print(importance_sorted, '\n \n \n', len(names))
    #plt.xticks(range(len(importance_sorted)), names, rotation='vertical')
    if mdi and config.TREE:
        plt.ylabel("Mean decrease in impurity [a.u.]")
    elif mda and config.TREE:
        plt.ylabel("Mean accuracy decrease [a.u.]")
    else: # svm
        plt.ylabel("Weights [a.u.]")
    if find_important_features:
        plt.xlabel("Feature")
    elif find_important_channels:
        plt.xlabel("Channel")

    plt.savefig(file_name)
    plt.show()
    plt.close()


def plot_important_features(importances_decending):
    plot_importances(importances_decending, config.FEATURE_IMPORTANCE_HIST, config.FEATURE_NAMES, find_important_channels=False, find_important_features=True)


def plot_important_channels():
    names = np.array([np.loadtxt(config.CHANNELS_FILE, dtype=str)])
    importances = json.load(open(config.IMPORTANCE_FILE_CHANNELS))
    plot_importances(importances, config.CHANNEL_IMPORTANCE_HIST, names, find_important_channels=True, find_important_features=False)


def plot_results(importances_file, y_test, predicted, find_important_features):
    importances = json.load(open(importances_file))
    importances_decending = dict(sorted(importances.items(), key=lambda x:abs(x[1]), reverse=True)) # abs value bc for svm the importances are both negative and positive 
    if find_important_features:
        confusion_matrix_heatmap(config.HEATMAP, y_test, predicted)
        plot_important_features(importances_decending)
        #plot_accuracies(importances_decending)
        
    else: #find_important_channels:
        plot_important_channels()
        
