from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit

models = {"Logistic Regression" : LogisticRegression(), 
            "Nearest Neighbors":KNeighborsClassifier() ,  
            "SVM": SVC(), 
            "Gaussian Process" : GaussianProcessClassifier(1.0 * RBF(1.0)),
            "Decision Tree" : DecisionTreeClassifier(), 
            "Random Forest": RandomForestClassifier(), 
            "Neural Net" : MLPClassifier(), 
            "AdaBoost" : AdaBoostClassifier()
            "Naive Bayes" : GaussianNB() , 
            "QDA": QuadraticDiscriminantAnalysis()
}

params = {'Logistic Regression':{'penalty': ['l1', 'l2'], 'C': [1, 10], 'class_weight':[{1: w} for w in [1, 5, 10, 100]] },
           "Nearest Neighbors":{'n_neighbors':[3,5,10], 'algorithm':['kd_tree']},
           "SVM": {'kernel': ['rbf'], 'class_weight':[{1: w} for w in [1,5]]},
           'Gaussian Process':{},
            "Decision Tree" : {"criterion": ["gini", "entropy"], 'class_weight':[{1: w} for w in [1, 5, 10, 100]] },
            'Random Forest': {'n_estimators': [16, 32], 'max_depth': [10, 50, None],
                              'max_features': ['sqrt'],
                             'class_weight':[{1: w} for w in [1, 2, 10, 100]] },
            "Neural Net" :{'activation': ['relu'],
                           'solver': ['adam'],
                           'alpha': [0.0001],
                           'learning_rate': ['adaptive']},
           'AdaBoost':  {}
            'Naive Bayes':{},
#            'QDA':{}
}

def custom_scorer(y_true, y_pred):
    return f1_score(y_true, y_pred)