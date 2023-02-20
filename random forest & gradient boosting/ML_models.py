from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split 
import time

import config

def support_vector_machine(X_train, X_test, y_train):
    '''
    Runs and fits the support cevtor machine classifier
    
        param X_train: training data, matrix with features as rows and samples as columns
        param X_test: test data, matrix with features as rows and samples as columns
        param y_train: labels for training set, list
        
        return clf: the fitted support vector machine, object
        return y_rf_predict_test: the predicted classes, list
    '''
    svmClf = svm.SVC(kernel=config.SVM_KERNEL)
    svmClf.fit(X_train, y_train)
    predicted = svmClf.predict(X_test)
    return svmClf, predicted

def random_forest(X_train, X_test, y_train):
    '''
    Runs and fits the random forest classifier
    
        param X_train: training data, matrix with features as rows and samples as columns
        param X_test: test data, matrix with features as rows and samples as columns
        param y_train: labels for training set, list
        
        return rf: the fitted random forest, object
        return y_rf_predict_test: the predicted classes, list
    '''
    
    rf = RandomForestClassifier(random_state=0)
    rf.fit(X_train, y_train)
    predicted = rf.predict(X_test)
    return rf, predicted


def gradient_boosting(learning_rate, X_train, X_test, y_train):
    """
    Runs and fits the gradient boosting classifier
    
        param learning_rate: shrinks the contribution of each tree, float
        param X_train: training data, matrix with features as rows and samples as columns, list of lists
        param X_test: test data, matrix with features as rows and samples as columns, list of lists
        param y_train: labels for training set, list
        
        return gb: the fitted gradient boosting, object
        return y_rf_predict_test: the predicted classes, list
    """
    gb = GradientBoostingClassifier(learning_rate=learning_rate)
    gb.fit(X_train, y_train)
    predicted = gb.predict(X_test)
    return gb, predicted

def classification(model, features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)
    start = time.time()
    if model == "rf": 
        ml_model, predicted = random_forest(X_train, X_test, y_train)
    elif model == "gb":
        ml_model, predicted = gradient_boosting(config.LEARNING_RATE, X_train, X_test, y_train)
    elif model =="svm":
        ml_model, predicted = support_vector_machine(X_train, X_test, y_train)
    end = time.time()
    print("ELAPSED TIME FOR ML MODEL", end - start)
    
    return ml_model, predicted, X_test, y_test

