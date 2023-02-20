import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def heat(model):
    plt.figure()
    plt.title("Model: " + str(model).split('(')[0])# + " Patient: " + str(config.start_patient) + " to " + str(config.end_patient))
    #data = {'y_Actual': target, 'y_Predicted': predicted}
    #dataset = pd.DataFrame(data, columns=['y_Predicted','y_Actual'])
    conf_matrix = [[int(4789*1.35), int(247*1.35)],[int(25*1.35), int(821*1.35)]]
    lang = ['False', 'True',]
    conf_matrix_df = pd.DataFrame(conf_matrix,columns=lang,index=lang)
    #c_matrix_l = [[6807, 37],[58, 1529]]#pd.crosstab(dataset['y_Predicted'], dataset['y_Actual'],
        #rownames=['Predicted'], colnames=['Actual'])
    
    sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.show()

heat("Support vector machine")

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

model = "svm"

#importances = {26: 0.046848982550607876, 28: 0.04369247575270807, 18: 0.04366492416588419, 32: 0.04352412041226539, 31: 0.04342108113688825, 29: 0.04292491114528996, 27: 0.04269352558923434, 30: 0.04159414525943994, 23: 0.041114566285787135, 19: 0.04104922649133991, 25: 0.04088746558871259, 21: 0.039636713758801734, 22: 0.03947855743587658, 24: 0.03938841787608771, 20: 0.03871093340213527, 9: 0.036413282649911505, 17: 0.03567736452525394, 15: 0.03563973907438689, 8: 0.0353614340840962, 5: 0.03525180548470597, 10: 0.03510810863441993, 13: 0.03471259420826833, 4: 0.03470368719351913, 6: 0.034698588570619915, 11: 0.034557154839155446, 14: 0.034530506404511385, 16: 0.03448352384350517, 1: 0.03402223468451382, 12: 0.0323714824267326, 3: 0.0316848080241221, 2: 0.03115778725608742, 7: 0.030078696694753493, 0: 0.02971419614775106}
importances = {18: -0.6412565695686528, 9: -0.6252972405680837, 1: 0.5496385518267504, 20: -0.509373318585868, 22: 0.37087706689720135, 7: 0.346116502643173, 23: -0.33044986315016756, 32: 0.29673920609562926, 17: -0.2719168209467708, 5: -0.24645881823186255, 24: 0.23569969231162816, 6: 0.23378226675640945, 14: -0.22983530465067714, 13: 0.22617342086398232, 31: 0.22512171735976397, 28: 0.21994430793008146, 15: -0.19170933333769424, 8: -0.17323732962880492, 21: 0.1640008447683485, 4: -0.13434473677124237, 16: 0.13354411479843917, 12: 0.11970218378868708, 19: -0.11854453080110888, 11: 0.10783871078603771, 29: -0.08707503512970567, 25: -0.08535496860150155, 2: 0.08413843941395734, 0: -0.08329978421367612, 10: -0.07958441692702338, 3: -0.05470152140612326, 30: 0.048910780212979905, 27: -0.04338339566078804, 26: -0.011383339346590764}

importances_decending = dict(sorted(importances.items(), key=lambda x:abs(x[1]), reverse=True)) # abs value bc for svm the importances are both negative and positive 
print(importances_decending)
accuracies = np.loadtxt("acc_" + str(model) + ".txt")
plot_accuracies(accuracies, importances_decending)
        


def plot_importance(importance_sorted, find_important_channels, find_important_features, mdi, mda, names, tree):
    print(len(names[0]))
    #if mdi: 
    plt.bar(range(len(importance_sorted)), list(importance_sorted.values()))
    #else: # mda
    #    plt.bar(importances, yerr=result.importances_std)#, ax=ax)
    plt.xticks(range(len(importance_sorted)), names, rotation='vertical')
    if mdi and tree:
        plt.ylabel("Mean decrease in impurity [a.u.]")
    elif mda and tree:
        plt.ylabel("Mean accuracy decrease [a.u.]")
    else: # svm
        plt.ylabel("Weights [a.u.]")
    if find_important_features:
        plt.xlabel("Feature")
    elif find_important_channels:
        plt.xlabel("Channel")

    #plt.savefig(file_name)
    plt.show()
    plt.close()

model = "svm"
# PLOT IMPORTANT feature
#names = np.array([np.loadtxt("names_Rfeature_channelTrueFalse", dtype=str)])
#np.loadtxt("importance_sorted_mdaFalsemdiTruetreeTruefeature_channelFalseFalse.txt")
###importances = {0: 0.0, 1: 0.8566900973541955, 2: 8.058790996347536e-13, 3: 0.0, 4: 8.952504824547216e-15, 5: 0.0010816462555513719, 6: 0.02296257955886895, 7: 0.0, 8: 1.182522723985125e-05, 9: 0.0, 10: 1.4570883225001602e-09, 11: 0.0, 12: 5.945568448353922e-07, 13: 0.0, 14: 4.012234507099549e-14, 15: 0.0, 16: 0.0, 17: 2.926817259066984e-08, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0, 24: 0.00023882773904785907, 25: 0.11403940482156485, 26: 1.3150738350979528e-07, 27: 1.979823864471875e-11, 28: 0.0049217719647061346, 29: 5.309026868286252e-05, 30: 0.0, 31: 0.0, 32: 0.0}

importances = {0: -0.08329978421367612, 1: 0.5496385518267504, 2: 0.08413843941395734, 3: -0.05470152140612326, 4: -0.13434473677124237, 5: -0.24645881823186255, 6: 0.23378226675640945, 7: 0.346116502643173, 8: -0.17323732962880492, 9: -0.6252972405680837, 10: -0.07958441692702338, 11: 0.10783871078603771, 12: 0.11970218378868708, 13: 0.22617342086398232, 14: -0.22983530465067714, 15: -0.19170933333769424, 16: 0.13354411479843917, 17: -0.2719168209467708, 18: -0.6412565695686528, 19: -0.11854453080110888, 20: -0.509373318585868, 21: 0.1640008447683485, 22: 0.37087706689720135, 23: -0.33044986315016756, 24: 0.23569969231162816, 25: -0.08535496860150155, 26: -0.011383339346590764, 27: -0.04338339566078804, 28: 0.21994430793008146, 29: -0.08707503512970567, 30: 0.048910780212979905, 31: 0.22512171735976397, 32: 0.29673920609562926}

#np.loadtxt("importance_sorted_mdaFalsemdiTruetreeFalsefeature_channelFalseTrue.txt")

namesR = np.array([np.loadtxt("names_Rfeature_channelFalseTrue", dtype=str)])
namesG = np.array([np.loadtxt("names_Gfeature_channelFalseTrue", dtype=str)])
namesS = ["FP1",
"F7",
"T7",
"P7",
"F3",
"C3",
"P3",
"FP2",
"F4",
"C4",
"P4",
"F8",
"T8",
"P8",
"FZ",
"CZ",
"FT9",
"FT10",
"O1",
"PZ",
"O2",
"C2",
"C6",
"CP2",
"CP4",
"CP6",
"01",
"FC1",
"FC2",
"FC5",
"FC6",
"CP1",
"CP5"]
#np.array([np.loadtxt("names_Sfeature_channelFalseTrue.txt", dtype=str)])

name = namesS
plot_importance(importances, find_important_channels=True, find_important_features=False, mdi=False, mda=True, names=name, tree=False)
