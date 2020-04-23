# Basic imports, Python's Built-in modules
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

from pprint import pprint
import time
import random
import urllib.request as req
import os
import sys
import time

# Import graph libraries.
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from mpl_toolkits import mplot3d

from ipywidgets import interact, fixed

# Import main modules, packages, and third party libraries.
from scipy import stats

import numpy as np; from numpy import nan
import pandas as pd
import seaborn as sns; sns.set()

# Import scikit-learn classes: datasets.
from sklearn.datasets import fetch_lfw_people                # Suitable Fetcher for Labeled Faces from Wild Dataset.
from sklearn.datasets import load_digits                     # Suitable Fetcher for Digits Dataset.

# Import scikit-learn classes: data generator classes.
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets.samples_generator import make_circles

# Import scikit-learn classes: models (Estimators).
from sklearn.naive_bayes import GaussianNB           # Non-parametric Generative Model
from sklearn.naive_bayes import MultinomialNB        # Non-parametric Generative Model
from sklearn.linear_model import LinearRegression    # Parametric Linear Discriminative Model
from sklearn.linear_model import LogisticRegression  # Parametric Linear Discriminative Model
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC                          # Parametric Linear Discriminative "Support Vector Classifier"
from sklearn.tree import DecisionTreeClassifier      # Non-parametric Model
from sklearn.ensemble import BaggingClassifier       # Non-parametric Model (Meta-Estimator, that is, an Ensemble Method)
from sklearn.ensemble import RandomForestClassifier  # Non-parametric Model (Meta-Estimator, that is, an Ensemble Method)

# Import scikit-learn classes: preprocessing step utility functions.
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA                # Unsupervised Machine Learning tasks: feature reduction, dimensionality reduction
from sklearn.mixture import GaussianMixture          # Unsupervised Machine Learning tasks: clustering
from sklearn.manifold import Isomap                  # Unsupervised Machine Learning tasks: feature reduction, dimensionality reduction
# Import scikit-learn classes: preprocessing.
from sklearn.preprocessing import PolynomialFeatures
# from sklearn.preprocessing import Imputer
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest

# Import scikit-learn classes: Hyperparameters Validation utility functions.
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import LeaveOneOut

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

# Import scikit-learn classes: model's evaluation step utility functions.
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Import scikit-learn classes: Transformers for basis functions.
from sklearn.base import BaseEstimator, TransformerMixin



def plot_confusion_matrix_sns(true_labels, predicted_labels, target_names, show_classification_report=False):
    """Plot confusion matrix of a trained model, also referred to as estimator, using seaborn graphics library.
    Params:
    -------
    	:true_labels: iterable object, suggested one-dimensional list or numpy.ndarray;
	:predicted_labels: iterable object, suggested one-dimensional list or numpy.ndarray;
	:target_names: iterable object, suggested one-dimensional list or numpy.ndarray of target's names, that are, category names;
        :show_classification_report: parameter with a function of flag value, default value set to False. If set to True the function will also print the classification report for inference via `sklearn.metrics.classification_report` function.
    """
    
    mat = confusion_matrix(true_labels, predicted_labels)
    sns.heatmap(mat.T, square=True, \
           annot=True, fmt='d', \
           cbar=False, \
           xticklabels=target_names, yticklabels=target_names, )
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    
    print( \
        classification_report(true_labels, predicted_labels)
    )
    pass
    
def example_random_forests_estimator(
    n_samples=1000, n_features=4,
    n_informative=2, n_redundant=0,
    random_state=0, shuffle=False,
    show_confusion_matrix=False,
    show_classification_report=True):
    """Brief example of usage of Random Forest Tecquinique, Classifier."""


    target_names = np.array(range(n_informative), dtype=np.int)
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=n_informative, n_redundant=n_redundant,
        random_state=random_state, shuffle=shuffle)
        
    Xtrain, Xtest, ytrain, ytest = \
        train_test_split(
            X, y,
            shuffle=True,
            random_state=random_state)
        
    clf = RandomForestClassifier(max_depth=2, random_state=random_state)
    clf.fit(Xtrain, ytrain)

    print("")
    print("Feature Importances:")
    print(clf.feature_importances_)

    print("")
    print("Predict:")
    samples = [np.zeros(n_features)]
    # print(clf.predict([[0, 0, 0, 0]]))
    print(clf.predict(samples))
    
    if show_confusion_matrix:
        print("")
        print("Plot Confusion Matrix:")
        ypred = clf.predict(Xtest)
        plot_confusion_matrix_sns(ytest, ypred, target_names, show_classification_report=show_classification_report)
    pass

def example_grid_search_random_forests_estimator(
    n_samples=1000, n_features=4,
    n_informative=2, n_redundant=0,
    random_state=0, shuffle=False,
    show_confusion_matrix=False,
    show_classification_report=True,
    cv=5):
    """Brief example of usage of Random Forest Tecquinique, Classifier."""

    print('Creating Data...')
    target_names = np.array(range(n_informative), dtype=np.int)
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=n_informative, n_redundant=n_redundant,
        random_state=random_state, shuffle=shuffle)
        
    Xtrain, Xtest, ytrain, ytest = \
        train_test_split(
            X, y,
            shuffle=True,
            random_state=random_state)
        
    param_grid = {
       'n_estimators': [100, 200, 300,],
       'criterion':  ['gini','entropy',],
       'n_jobs': [3],
       'max_features': [int, float, None, 'sqrt', 'log2'],
       'bootstrap': [True, False]
    }
    clf = RandomForestClassifier()
    grid = GridSearchCV(
       estimator=clf, param_grid=param_grid,
       cv=cv, verbose=0)
    
    print('Training step is running...')
    clf.fit(Xtrain, ytrain)

    print("")
    print("Feature Importances:")
    print(clf.feature_importances_)

    print("")
    print("Predict:")
    samples = [np.zeros(n_features)]
    # print(clf.predict([[0, 0, 0, 0]]))
    print(clf.predict(samples))
    
    if show_confusion_matrix:
        print("")
        print("Plot Confusion Matrix:")
        ypred = clf.predict(Xtest)
        plot_confusion_matrix_sns(ytest, ypred, target_names, show_classification_report=show_classification_report)
    pass
