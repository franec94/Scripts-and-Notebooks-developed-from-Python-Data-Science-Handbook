# Matplotlib pyplot provides plotting API
import matplotlib as mpl
from matplotlib import pyplot as plt
import chart_studio.plotly.plotly as py

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler # Standardize data (0 mean, 1 stdev)
from sklearn.preprocessing import Normalizer     # Normalize data (length of 1)
from sklearn.preprocessing import Binarizer      # Binarization

def preprocess_categorical_variables(df, columns_2_avoid=None):
    feature_vs_values = dict()

    if columns_2_avoid is not None:
        columns_2_keep = list(filter(lambda x: x not in columns_2_avoid, df.columns))
    else:
        columns_2_keep = df.columns
    
    # for index, predictor in enumerate(df.columns):
    for _, predictor in enumerate(columns_2_keep):
        # print(index, predictor)
       labels = df[predictor].astype('category').cat.categories.tolist()
       # print(labels)
       replace_map_comp = {predictor : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
       # print(replace_map_comp)
       df.replace(replace_map_comp, inplace=True)

       feature_vs_values[predictor] = replace_map_comp[predictor]
    # print(feature_vs_values)

    return feature_vs_values

def preprocessing_data_rescaling(rescaling_method='minmax', X=None):
    if X is None:
        raise Exception("[ERROR] - data to be preprocessed was passed as NoneType which si forbidden")
    if rescaling_method == 'standard':
        # Fit on training set only.
        scaler = StandardScaler().fit(X)
        rescaledX = scaler.transform(X)
    elif rescaling_method == 'norm':
        scaler = Normalizer().fit(X)
        rescaledX = scaler.transform(X)
    elif rescaling_method == 'minmax':
        scaler = MinMaxScaler().fit(X)
        rescaledX = scaler.transform(X)
    else:
        raise Exception(f"[ERROR] - Preprocessing method `{rescaling_method}` not yet supported or just not allowed")

    # rescaledX = preprocessing.normalize(X)
    
    # print('shape features matrix X, after normalizing: ', X.shape)
    print('shape features matrix X, after normalizing: ', rescaledX.shape)
    
    return rescaledX

def preprocessing_stage_dataset(df, dict_preprocessing=None):
    return df
