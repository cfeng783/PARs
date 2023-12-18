'''
Created on Aug 10, 2023

@author: Cheng Feng
'''
from .features import NumericFeature,CategoricFeature
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class DataUtil():
    '''
    Utility class for data processing
    
    Parameters
    ----------
    features : list, default is None
        the list of features, must be specified when filename is None
    filename : string, default is None
        the path of json file to load DataUtil, must be specified when features is None
    '''
    def __init__(self, features=None, filename=None):
        if filename is None:
            self.features = features
        else:
            self._load_jsonfile(filename)
        
        self.feature_map = {feature.name: feature for feature in self.features}
    
    def onehot_encode(self, df):
        """
        onehot encode the features in the dataset
        
        Parameters
        ----------
        df : DataFrame
            The dataset
            
        Returns
        -------
        Dataframe 
            the modified dataframe
        """
        df = df.copy()
        #'onehot encoding'
        cols = []
        for feature in self.features:
            if isinstance(feature, CategoricFeature):
                for value in feature.values:
                    new_entry = feature.get_feature_name(value)
                    df[new_entry] = 0                   
                    df.loc[df[feature.name]==value,new_entry] = 1
                    cols.append(new_entry)
            if isinstance(feature, NumericFeature):
                cols.append(feature.name)
        df = df[cols]
        return df
    
    def save2jsonfile(self,filename):
        """
        save to a Json file
        
        Parameters
        ----------
        filename : string
            the path of json file to save DataUtil
             
        """
        ret = {}
        ret['features'] = []
        for feature in self.features:
            ret['features'].append(feature.to_dict())
        with open(filename, 'w') as outfile:
            json.dump(ret,outfile,cls=NpEncoder)
    
    def _load_jsonfile(self,filename):
        with open(filename, 'r') as inputfile:
            data = inputfile.read()
        configJson = json.loads(data)
        self.features = []
        for sd in configJson['features']:            
            name = sd['name']
            dtype = sd['dtype']
            if dtype == 'numeric':
                feature = NumericFeature(name,min_value=sd['min_value'], max_value=sd['max_value'],
                                            mean_value=sd['mean_value'], std_value=sd['std_value'])
            elif dtype == 'categoric':
                feature = CategoricFeature(name, values=sd['values'])
            self.features.append(feature)