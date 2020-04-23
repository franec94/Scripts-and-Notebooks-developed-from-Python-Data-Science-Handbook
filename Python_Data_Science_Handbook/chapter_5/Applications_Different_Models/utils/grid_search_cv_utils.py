# Magic statements.

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

from pprint import pprint
import time

# Import graph libraries.
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# Import main modules, packages, and third party libraries.
import numpy as np; from numpy import nan
import pandas as pd
import seaborn as sns; sns.set()

# Import scikit-learn classes: datasets.
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_digits
from sklearn.datasets import load_iris

# Import scikit-learn classes: preprocessing step utility functions.
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA                # Unsupervised Machine Learning tasks: feature reduction, dimensionality reduction
from sklearn.mixture import GaussianMixture          # Unsupervised Machine Learning tasks: clustering
from sklearn.manifold import Isomap                  # Unsupervised Machine Learning tasks: feature reduction, dimensionality reduction

# Import scikit-learn classes: models (Estimators).
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge

# Import scikit-learn classes: preprocessing.
from sklearn.preprocessing import PolynomialFeatures
# from sklearn.preprocessing import Imputer
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle

from sklearn.neighbors import KNeighborsClassifier

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


def decorator_run_pipeline_cv(
    a_func
  ):
    
    def wrapper_decorator(
      model=None, train=None,
      test=None, random_state=0, cv=None,
    ):
        assert model is not None, 'model is None'
        assert train is not None, 'train is None'
        assert test is not None, 'test is None'
    
        print('[*] Running pipeline...')

        a_func(model, train, test, random_state, cv)
    
        print('[*] Pipeline done.')
        pass

    return wrapper_decorator

@decorator_run_pipeline_cv
def run_pipeline(model, train, test, random_state=0, cv=None):
    
    # Shuffle data used for training.
    Xtrain, ytrain = train.data, train.target
    Xtrain, ytrain = shuffle(Xtrain, ytrain, random_state=random_state)

    # Shuffle data used for test phase, if any.
    Xtest, ytest = test.data, test.target
    Xtest, ytest = shuffle(Xtest, ytest, random_state=random_state)

    # check whether to perform cv, having provided as input argument
    # passed to this function a quantity representing:
    # - either, number of folds in which training set will be splitted
    # - or, a cross-validation scheme, techique, pattern, represented
    #   by means of a Scikit-Learn class object.
    if cv is not None:
        print('[*] CV running...')
        scores = cross_val_score(model, Xtrain, ytrain , cv=cv)
        print(scores)
        print(scores.mean())
        print('[*] CV done.')
    
    # Fit the model to training data
    model.fit(Xtrain, ytrain)
    labels = model.predict(Xtest)
    
    if test is not None:
        mat = confusion_matrix(ytest, labels)
        sns.heatmap(mat.T, square=True,
             annot=True, fmt='d',
             cbar=False,
             xticklabels=train.target_names, yticklabels=train.target_names, )
        plt.xlabel('true label')
        plt.ylabel('predicted label')

        print('K-Neighbors Classifier accuracy score:', accuracy_score(ytest, labels))
        print(f"K-Neighbors Classifier accuracy score (percentage): {accuracy_score(ytest, labels)*100:.2f}%")
    
    def predict_category(s, train=train, model=model):
        pred = model.predict([s])
        
        return ', '.join([ str(pred), str(train.target_names[pred[0]]) ])
    
    print(predict_category('sending a payload to the ISS'))
    print(predict_category('discussing islam versus atheism'))
    print(predict_category('determinig screen resolution and size'))
    return model

def decorator_inner_vs_outer_cv(
    a_func
  ):
    def wrapper_decorator(
      clf,
      Xtrain, ytrain,
      param_grid,
      Xtest, ytest,
      num_trials=30,
      random_state=0,
      verbose=0,
    ):

        assert clf is not None, 'cls is None'
        assert train is not None, 'train is None'
        assert param_grid is not None, 'param_grid is None'

        print('[*] inner_vs_outer_cv running...')
        a_func(
          clf,
          Xtrain, ytrain,
          param_grid,
          Xtest, ytest,
          test=None,
          num_trials=30,
          random_state=0,
          verbose=0
        )
        print('[*] inner_vs_outer_cv done.')
        pass
    return wrapper_decorator

@decorator_inner_vs_outer_cv
def inner_vs_outer_cv(
    clf,
    Xtrain, ytrain,
    param_grid,
    Xtest, ytest,
    test=None,
    num_trials=30,
    random_state=0,
    verbose=0):
  
    Xtrain, ytrain = shuffle(Xtrain, ytrain, random_state=random_state)

    if Xtest is not None and ytest is not None:
        Xtest, ytest = shuffle(Xtest, ytest, random_state=random_state)

    # Arrays to store scores
    non_nested_scores = np.zeros(num_trials)
    nested_scores = np.zeros(num_trials)

    # Loop for each trial
    for i in range(num_trials):
        # Choose cross-validation techniques for the inner and outer loops,
        # independently of the dataset.
        # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
        inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
        outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)

        # Non_nested parameter search and scoring
        clf = GridSearchCV(estimator=lrm, param_grid=param_grid, cv=inner_cv)
        clf.fit(Xtrain, ytrain)
        non_nested_scores[i] = clf.best_score_

        # Nested CV with parameter optimization
        nested_score = cross_val_score(clf, X=Xtrain, y=ytrain, cv=outer_cv)
        nested_scores[i] = nested_score.mean()

    score_difference = non_nested_scores - nested_scores

    print("Average difference of {:6f} with std. dev. of {:6f}."
        .format(score_difference.mean(), score_difference.std()))

    # Plot scores on each trial for nested and non-nested CV
    plt.figure()
    plt.subplot(211)
    non_nested_scores_line, = plt.plot(non_nested_scores, color='r')
    nested_line, = plt.plot(nested_scores, color='b')
    plt.ylabel("score", fontsize="14")
    plt.legend([non_nested_scores_line, nested_line],
             ["Non-Nested CV", "Nested CV"],
            bbox_to_anchor=(0, .4, .5, 0))
    plt.title("Non-Nested and Nested Cross Validation on Iris Dataset",
            x=.5, y=1.1, fontsize="15")

    # Plot bar chart of the difference.
    plt.subplot(212)
    difference_plot = plt.bar(range(NUM_TRIALS), score_difference)
    plt.xlabel("Individual Trial #")
    plt.legend([difference_plot],
           ["Non-Nested CV - Nested CV Score"],
           bbox_to_anchor=(0, 1, .8, 0))
    plt.ylabel("score difference", fontsize="14")

    plt.show()
    pass

def load_dataset_bridges_pittsburg():
    