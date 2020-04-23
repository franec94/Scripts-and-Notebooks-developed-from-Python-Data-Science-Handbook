# Import graph libraries.
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# Import main modules, packages, and third party libraries.
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()

# Import scikit-learn classes: datasets.
from sklearn.datasets import load_iris

# Import scikit-learn classes: preprocessing step utility functions.
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA # Unsupervised Machine Learning tasks: feature reduction, dimensionality reduction
from sklearn.mixture import GaussianMixture # Unsupervised Machine Learning tasks: clustering
from sklearn.manifold import Isomap # Unsupervised Machine Learning tasks: feature reduction, dimensionality reduction

# Import scikit-learn classes: models
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

# Import scikit-learn classes: Hyperparameters Validation utility functions.
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import LeaveOneOut

# === UTILS IMPORTS (Done by myself) ==== #
from utils.display_utils import display_heatmap
from utils.display_utils import show_frequency_distribution_predictors
from utils.display_utils import show_categorical_predictor_values
from utils.display_utils import  show_cum_variance_vs_components

from utils.preprocessing_utils import preprocess_categorical_variables
from utils.preprocessing_utils import  preprocessing_data_rescaling

from utils.training_utils import sgd_classifier_grid_search
from utils.training_utils import naive_bayes_classifier_grid_search
from utils.training_utils import svm_linear_classifier_grid_search
from utils.training_utils import decision_tree_classifier_grid_search
from utils.training_utils import random_forest_classifier_grid_search
from utils.training_utils import plot_roc_crossval

def load_brdiges_dataset():
    dataset = None
    
    dataset_path = '/home/franec94/Documents/datasets/datasets_folders/pittsburgh-bridges-data-set'
    dataset_name = 'bridges.data.csv'

    # column_names = ['IDENTIF', 'RIVER', 'LOCATION', 'ERECTED', 'PURPOSE', 'LENGTH', 'LANES', 'CLEAR-G', 'T-OR-D', 'MATERIAL', 'SPAN', 'REL-L', 'TYPE']
    column_names = ['RIVER', 'LOCATION', 'ERECTED', 'PURPOSE', 'LENGTH', 'LANES', 'CLEAR-G', 'T-OR-D', 'MATERIAL', 'SPAN', 'REL-L', 'TYPE']
    dataset = pd.read_csv('{}/{}'.format(dataset_path, dataset_name), names=column_names, index_col=0)
    
    print('Dataset shape: {}'.format(dataset.shape))
    print(dataset.info())
    
    # === DISCOVERING VALUES WITHIN EACH PREDICTOR DOMAIN === #
    columns_2_avoid = ['ERECTED', 'LENGTH', 'LOCATION', 'LANES']
    # columns_2_avoid = None
    list_columns_2_fix = show_categorical_predictor_values(dataset, columns_2_avoid)
    
    # === FIXING, UPDATING NULL VALUES CODED AS '?' SYMBOL  === #
    # === WITHIN EACH CATEGORICAL VARIABLE, IF DETECTED ANY === #
    print('Before', dataset.shape)
    for _, predictor in enumerate(list_columns_2_fix):
        dataset = dataset[dataset[predictor] != '?']
    print('After', dataset.shape)

    _ = show_categorical_predictor_values(dataset, columns_2_avoid)
    
    
    # === INTERMEDIATE RESULT FOUNDED === #
    preprocess_categorical_variables(dataset, columns_2_avoid)
    print(dataset.info())
    
    # === MAP NUMERICAL VALUES TO INTEGER VALUES === #
    print('Before', dataset.shape)
    columns_2_map = ['ERECTED', 'LANES']
    for _, predictor in enumerate(columns_2_map):
        dataset = dataset[dataset[predictor] != '?']
        dataset[predictor] = np.array(list(map(lambda x: int(x), dataset[predictor].values)))
    print('After', dataset.shape)
    print(dataset.info())
    print(dataset.head(5))
    
    # === MAP NUMERICAL VALUES TO FLOAT VALUES === #
    print('Before', dataset.shape)
    columns_2_map = ['LOCATION', 'LANES', 'LENGTH']    
    for _, predictor in enumerate(columns_2_map):
        dataset = dataset[dataset[predictor] != '?']
        dataset[predictor] = np.array(list(map(lambda x: float(x), dataset[predictor].values)))
    print('After', dataset.shape)    
    print(dataset.info())
    print(dataset.head(5))

    # columns_2_avoid = None
    list_columns_2_fix = show_categorical_predictor_values(dataset, None)
    
    result = dataset.isnull().values.any()
    # print('After handling null values\nThere are any null values ? Response: {}'.format(result))

    result = dataset.isnull().sum()
    # print('Number of null values for each predictor:\n{}'.format(result))
    print(dataset.describe(include='all'))
    return dataset