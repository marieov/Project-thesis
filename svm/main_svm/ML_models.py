from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm

def support_vector_machine(X_train, X_test, y_train, y_test):
    '''
    Runs and fits the support cevtor machine classifier
    
        param X_train: training data, matrix with features as rows and samples as columns
        param X_test: test data, matrix with features as rows and samples as columns
        param y_train: labels for training set, list
        param y_test: labels for test set, list
        
        return clf: the fitted support vector machine, object
        return y_rf_predict_test: the predicted classes, list
    '''
    print("111")
    svmClf = svm.SVC(kernel='linear') #TODO prøv med flere # probability=Ture to avoid beeing stuck in local minima
    print("222")
    svmClf.fit(X_train, y_train)
    print("333")
    predicted = svmClf.predict(X_test)
    print("444")
    return svmClf, predicted

def random_forest(X_train, X_test, y_train, y_test):
    '''
    Runs and fits the random forest classifier
    
        param X_train: training data, matrix with features as rows and samples as columns
        param X_test: test data, matrix with features as rows and samples as columns
        param y_train: labels for training set, list
        param y_test: labels for test set, list
        
        return rf: the fitted random forest, object
        return y_rf_predict_test: the predicted classes, list
    '''
    
    rf = RandomForestClassifier(random_state=0)
    rf.fit(X_train, y_train)
    predicted = rf.predict(X_test)
    return rf, predicted


def gradient_boosting(learning_rate, X_train, X_test, y_train, y_test):
    """
    Runs and fits the gradient boosting classifier
    
        param learning_rate: shrinks the contribution of each tree, float
        param X_train: training data, matrix with features as rows and samples as columns, list of lists
        param X_test: test data, matrix with features as rows and samples as columns, list of lists
        param y_train: labels for training set, list
        param y_test: labels for test set, list
        
        return gb: the fitted gradient boosting, object
        return y_rf_predict_test: the predicted classes, list
    """
    gb = GradientBoostingClassifier(learning_rate=learning_rate)
    gb.fit(X_train, y_train)
    predicted = gb.predict(X_test)
    return gb, predicted

