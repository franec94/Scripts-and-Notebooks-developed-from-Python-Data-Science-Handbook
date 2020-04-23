# Standard Imports (Data Manipulation and Graphics)
import numpy as np    # Load the Numpy library with alias 'np' 
import pandas as pd   # Load the Pandas library with alias 'pd' 

import seaborn as sns # Load the Seabonrn, graphics library with alias 'sns' 

import copy
from scipy import stats
from scipy import interp

# Matplotlib pyplot provides plotting API
import matplotlib as mpl
from matplotlib import pyplot as plt
import chart_studio.plotly.plotly as py

# Preprocessing Imports
# from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler # Standardize data (0 mean, 1 stdev)
from sklearn.preprocessing import Normalizer     # Normalize data (length of 1)
from sklearn.preprocessing import Binarizer      # Binarization

# Imports for handling Training
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

# After Training Analysis Imports
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

# Classifiers Imports
# SVMs Classifieres
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn import svm

# Bayesian Classifieres
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

# Decision Tree Classifieres
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier



def get_classifier(type_classifier, params_classifier):
    if type_classifier == 'sgd':
        clf = SGDClassifier(loss=params_classifier.best_params_['clf__loss'], penalty=params_classifier.best_params_['clf__penalty'], \
            alpha=params_classifier.best_params_['clf__alpha'], random_state=70, \
            max_iter=params_classifier.best_params_['clf__max_iter'], tol=None)
    elif type_classifier == 'linear-svm':
        clf = LinearSVC( \
            loss=params_classifier.best_params_['clf__loss'],
            penalty=params_classifier.best_params_['clf__penalty'], \
            C=params_classifier.best_params_['clf__C'], \
            random_state=70, max_iter=50, tol=None)

    elif type_classifier == 'rbf-svm':
        # print(params_classifier.best_params_)
        clf = svm.SVC( \
                kernel=str(params_classifier.best_params_['clf__kernel']), 
                max_iter=int(params_classifier.best_params_['clf__max_iter']), 
                # gamma=float(params_classifier.best_params_['clf__gamma']), \
                gamma=float(0.001),
                C=float(params_classifier.best_params_['clf__C']), \
                random_state=int(70), tol=None)
    elif type_classifier == 'decision-tree':
        clf = DecisionTreeClassifier(random_state=70,
                                     splitter=params_classifier.best_params_['clf__splitter'],
                                     criterion=params_classifier.best_params_['clf__criterion'],
                                     max_features=params_classifier.best_params_['clf__max_features'],
                                    )
    elif type_classifier == 'random-forest':
        clf = RandomForestClassifier(random_state=70,
                                     n_estimators=params_classifier.best_params_['clf__n_estimators'],
                                     criterion=params_classifier.best_params_['clf__criterion'],
                                     bootstrap=params_classifier.best_params_['clf__bootstrap'],
                                    )
    else:
        raise Exception('Error {}'.format(type_classifier))

    print(clf)
    return clf

def evaluate_best_current_model_(X, y, pca, gs_clf, test_size, random_state, type_classifier):
    X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = train_test_split(
                X, y, test_size=50, random_state=random_state)
    X_train_pca_tmp = pca.transform(X_train_tmp)
            
    tmp_clf = get_classifier(type_classifier, gs_clf)

    tmp_clf_trained = tmp_clf.fit(X_train_pca_tmp, y_train_tmp)
    tmp_predicted = tmp_clf_trained.predict(pca.transform(X_test_tmp))

    print("[TRAINING WITH BEST MODEL]--- Classification Report ---")
    print(metrics.classification_report(y_test_tmp, tmp_predicted,
                                                target_names=['negative', 'positive']))

    print("[TRAINING WITH BEST MODEL]--- Confusion Matrix ---")
    print(metrics.confusion_matrix(y_test_tmp, tmp_predicted))
    print(f"{np.mean(tmp_predicted == y_test_tmp)}")
    pass

def grid_search_approach(technique, n, clf, parameters, X, y, test_size, random_state, cv=7, iid=False, n_jobs=-1, sss_flag=False, type_classifier=None):
    '''Performs grid search technique, against a defined classifier or pipeline object and a dictionary of hyper-params.
    
    Params:
    -------
        - n: number or list of numbers, so numbers of principal components to be retained, exploited,
             in order to improve the overall performances.
        
        - clf: scikit-learn Pipeline object, made up of all the operations to be performed in a given order.
        
        - cv: integer, default=7, number to refer to attempt performed by cross-validation technique to create
              cv models picking up their mean.
        
        - iid: boolean, default=False, shows whether input data should be treated as independent and
               identically distributed data samples.
        
        - n_jobs: integer, default=-1, allows, or enables to let the work station within which the training script is lauched to discover
                  and eventually exploit a baunch of cpu for increasing the performance during training phase.
    '''
    # === Splitting dataset into training and test sets, respectively, both features and labels === #
    if sss_flag is False:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        sss.get_n_splits(X, y)

        for train_index, test_index in sss.split(X, y):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        
        # sss = StratifiedShuffleSplit(n_splits=cv, test_size=test_size * 2, random_state=random_state)
        # cv = sss.split(X_train, y_train)

    # === Performing only once for all Principal Component === #
    n_components = X.shape[1]
    pca = PCA(n_components=n_components)
    
    pca = pca.fit(X_train)
    backup_pcs_ = copy.copy(pca.components_)
    # print(f"Shape principal componets: {backup_pcs_.shape}")
    
    print(f'==== GRID SEARCH METHOD APPLYED ON: {technique.split(",")[0]} Technique ====')
    print(f'==== PREPROCESSING METHOD: {technique.split(",")[1]} Technique ====')
    for pos, n_components in enumerate(n):
        print('\n', "*" * 20, sep='')
        print(f"Grid Search attempt no. : {pos+1}")
        tmp_cv = cv
        
        if True:
            # === Preparing Feature Space by means of retained Principal Components === #
            n = len(pca.components_[n_components:])    
            pca.components_[n_components:] = [[0] * X.shape[1]] * n
        
            X_train_pca = pca.transform(X_train)

            if sss_flag is True:
                sss = StratifiedShuffleSplit(n_splits=cv, test_size=test_size * 0.5, random_state=random_state)
                cv = sss.split(X_train, y_train)
        
            # === Performing training phase === #
            gs_clf = GridSearchCV(clf, parameters, cv=cv, iid=iid, n_jobs=n_jobs)
            gs_clf = gs_clf.fit(X_train_pca, y_train)
        
        
            # === Evaluating performances === #
            predicted = gs_clf.predict(pca.transform(X_test))
    
            print("--- Classification Report ---")
            print(metrics.classification_report(y_test, predicted,
                                                target_names=['negative', 'positive']))

            print("--- Confusion Matrix ---")
            print(metrics.confusion_matrix(y_test, predicted))
            print(f"{np.mean(predicted == y_test)}")
    
            print(f"Best Score: {gs_clf.best_score_}")

            print("--- Best Params ---")
            print(f"n_components: {n_components}")
            for param_name in sorted(parameters.keys()):
                print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
        
            try:
                evaluate_best_current_model_(X, y, pca, gs_clf, test_size, random_state, type_classifier)
                # raise Exception('Ok')
            except Exception as err:
                print(err)
            # evaluate_best_current_model_(X, y, pca, gs_clf, test_size, random_state, type_classifier)
            # === Restoring overall pcs for further, subsequent evaluation === #
            pca.components_ = copy.copy(backup_pcs_)
            cv = tmp_cv
            
        # except Exception as err:
        else: pass #print(err)
    pass

def sgd_classifier_grid_search(X, y):
    test_size, random_state = 0.25, 50

    # Pipeline classifier definition
    clf_sgd = Pipeline([
        ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=70,
                          max_iter=50, tol=None)),
    ])

    # Hyper-params definition, dictionary containing possible values established
    parameters_sgd_classifier = {
        'clf__loss': ('hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'),
        'clf__penalty': ('l2', 'l1', 'elasticnet'),
        'clf__alpha': (1e-1, 1e-2, 1e-3, 1e-4),
        'clf__max_iter': (50, 100, 150, 200, 500, 1000, 1500, 2000, 2500),
        'clf__learning_rate': ('optimal',),
        'clf__tol': (None, 1e-2, 1e-4, 1e-5, 1e-6),
    }

    # Defining number of principal components to be kept, retained for hyper-params selection
    # n = [int(xi) for xi in '2,5,6,7,8,9,10'.split(',')]
    n = [int(xi) for xi in '6,7,8,9,10'.split(',')]

    # Run the grid-search technique
    type_classifier = 'sgd'
    grid_search_approach('SGDClassifier,MinMax', n, clf_sgd, \
        parameters_sgd_classifier, X, y, \
        test_size, random_state, sss_flag=False, \
        type_classifier=type_classifier)
    pass

def plot_roc_crossval(X, y):
    n_samples, n_features = X.shape

    # Add noisy features
    random_state = np.random.RandomState(0)
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # #############################################################################
    # Classification and ROC analysis

    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=6)
    classifier = svm.SVC(kernel='linear', probability=True,
                         random_state=random_state)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    pass

def svm_linear_classifier_grid_search(X, y, kernel_type=None):
    test_size, random_state = 0.25, 50

    # Pipeline classifier definition
    if kernel_type is None:
        clf_svm = Pipeline([
            ('clf', LinearSVC(loss='squared_hinge', penalty='l2', C=1.0,
                random_state=70, max_iter=50, tol=None)),
        ])
         # Hyper-params definition, dictionary containing possible values established
        parameters_svm = {
            # 'clf__loss': ('hinge', 'squared_hinge'),
            'clf__penalty': ('l2','l1'),
            'clf__C': (1.0, 0.1, 0.001, 0.0001, 10.0),
            # 'clf__max_iter': (50, 100, 150, 200, 500, 1000, 1500, 2000),
            # 'clf__tol': (1e-2, 1e-4, 1e-5, 1e-6),
            # 'clf_dual': (True, False)
        }
        type_classifier = 'linear-svm'
        kernel_type = ''
    
    elif kernel_type == 'svm-rbf-kernel':
        clf_svm = Pipeline([
            ('clf', svm.SVC()),
        ])
        # Hyper-params definition, dictionary containing possible values established
        parameters_svm = {
            'clf__gamma': (0.003, 0.03, 0.05, 0.5, 0.7, 1.0, 1.5),
            'clf__max_iter':(1e+2, 1e+3, 2 * 1e+3, 5 * 1e+3, 1e+4, 1.5 * 1e+3),
            'clf__C': (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10, 1e+2, 1e+3),
        }
        parameters_svm['clf__kernel'] = ('rbf','linear')
        kernel_type = 'RBF_SVM'
        type_classifier = 'rbf-svm'
    else:
        raise Exception('Error')
    
    # Defining number of principal components to be kept, retained for hyper-params selection
    n = [int(xi) for xi in '2,5,6,7,8,9,10'.split(',')]

    # Run the grid-search technique
    grid_search_approach('{}_Classifier,MinMax'.format(kernel_type), n, clf_svm, \
        parameters_svm, X, y, \
        test_size, random_state, sss_flag=False, \
        type_classifier=type_classifier)
    pass

def naive_bayes_classifier_grid_search(X, y):
    
    type_classifier = 'naive-bayes'
    test_size, random_state = 0.20, 50
    
    # Defining number of principal components to be kept, retained for hyper-params selection
    n = [int(xi) for xi in '2,5,6,7,8,9,10'.split(',')]
    
    clf_naive_bayes = Pipeline([
        ('clf', GaussianNB()),
    ])
    
    parmas_naive_bayes = {}
    
    # Run the grid-search technique
    grid_search_approach('{}_Classifier,MinMax'.format('Naive_Bayes'), n, clf_naive_bayes, \
        parmas_naive_bayes, X, y, \
        test_size, random_state, sss_flag=False, \
        type_classifier=type_classifier)
    pass

def decision_tree_classifier_grid_search(X, y):
    
    type_classifier = 'decision-tree'
    test_size, random_state = 0.25, 50
    
    # Defining number of principal components to be kept, retained for hyper-params selection
    n = [int(xi) for xi in '2,5,6,7,8,9,10'.split(',')]
    
    clf_random_forest = Pipeline([
        ('clf', DecisionTreeClassifier(random_state=random_state)),
    ])
    
    parmas_random_forest = {
        'clf__splitter': ('random', 'best'),
        'clf__criterion':('gini', 'entropy'),
        'clf__max_features': (None, 'auto', 'sqrt', 'log2')
    }
    
    # Run the grid-search technique
    grid_search_approach('{}_Classifier,MinMax'.format('Random_Forest'), n, clf_random_forest, \
        parmas_random_forest, X, y, \
        test_size, random_state, sss_flag=False, \
        type_classifier=type_classifier)
    pass

def random_forest_classifier_grid_search(X, y):
    
    type_classifier = 'random-forest'
    test_size, random_state = 0.15, 50
    
    # Defining number of principal components to be kept, retained for hyper-params selection
    n = [int(xi) for xi in '2,5,6,7,8,9,10'.split(',')]
    
    clf_random_forest = Pipeline([
        ('clf', RandomForestClassifier(random_state=random_state)),
    ])
    
    parmas_random_forest = {
        'clf__n_estimators': (3, 5, 7, 10, 30, 50, 70, 100, 150, 200),
        'clf__criterion':('gini', 'entropy'),
        'clf__bootstrap': (True, False)
    }
    
    # Run the grid-search technique
    grid_search_approach('{}_Classifier,MinMax'.format('Random_Forest'), n, clf_random_forest, \
        parmas_random_forest, X, y, \
        test_size, random_state, sss_flag=False, \
        type_classifier=type_classifier)
    pass