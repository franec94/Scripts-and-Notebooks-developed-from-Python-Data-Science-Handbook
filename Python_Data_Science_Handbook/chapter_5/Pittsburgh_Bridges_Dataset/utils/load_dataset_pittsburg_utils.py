import pandas as pd
import numpy as np

# Matplotlib pyplot provides plotting API
import matplotlib as mpl
from matplotlib import pyplot as plt
import chart_studio.plotly.plotly as py
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from utils.display_utils import display_heatmap
from utils.display_utils import show_frequency_distribution_predictors
from utils.display_utils import show_categorical_predictor_values
from utils.display_utils import  show_cum_variance_vs_components

from utils.preprocessing_utils import preprocess_categorical_variables
from utils.preprocessing_utils import  preprocessing_data_rescaling

def load_pittsburg_dataset(describe_flag=False):
    """Utility function for loading pittsburg bridges dataset."""

    # Dataset location(path) and name:
    dataset_path = '/home/franec94/Documents/datasets/datasets_folders/pittsburgh-bridges-data-set'
    dataset_name = 'bridges.data.csv'

    # Loading dataset from path, plus name, both specified above, into a pandas dataframe:
    # column_names = ['IDENTIF', 'RIVER', 'LOCATION', 'ERECTED', 'PURPOSE', 'LENGTH', 'LANES', 'CLEAR-G', 'T-OR-D', 'MATERIAL', 'SPAN', 'REL-L', 'TYPE']
    column_names = ['RIVER', 'LOCATION', 'ERECTED', 'PURPOSE', 'LENGTH', 'LANES', 'CLEAR-G', 'T-OR-D', 'MATERIAL', 'SPAN', 'REL-L', 'TYPE']
    dataset = pd.read_csv('{}/{}'.format(dataset_path, dataset_name), names=column_names, index_col=0)

    columns_2_avoid = ['ERECTED', 'LENGTH', 'LOCATION', 'LANES']

    # Skip rows with N/A values coded as '?' symbol
    for _, predictor in enumerate(dataset.columns):
        dataset = dataset[dataset[predictor] != '?']
    
    # Mapping qualitaty variables to a integer range of values, for each categorica feature within the dataset:
    features_vs_values = preprocess_categorical_variables(dataset, columns_2_avoid)
    
    # Casting to integer value type features showing a numerical quantity as intger type
    columns_2_map = ['ERECTED', 'LANES']
    for _, predictor in enumerate(columns_2_map):
        dataset = dataset[dataset[predictor] != '?']
        dataset[predictor] = np.array(list(map(lambda x: int(x), dataset[predictor].values)))
    
    # Casting to integer value type features showing a numerical quantity as float type
    columns_2_map = ['LOCATION', 'LANES', 'LENGTH']   
    for _, predictor in enumerate(columns_2_map):
        dataset = dataset[dataset[predictor] != '?']
        dataset[predictor] = np.array(list(map(lambda x: float(x), dataset[predictor].values)))
    
    if describe_flag is True:
        # Display dataset major information
        print(dataset.describe(include='all'))
        print('Dataset shape: {}'.format(dataset.shape))
        print(dataset.info())
    
    return dataset, features_vs_values