'''
Created on Aug 10, 2023

@author: Cheng Feng
'''

import logging

logger = logging.getLogger(__name__)


class BaseFeature(object):
    '''
    The base feature class
    
    Parameters
    ----------
    name : string
        the name of the feature
    '''


    def __init__(self, name):
        self.name = name
    
    def to_dict(self):
        return {'name':self.name}


class NumericFeature(BaseFeature):
    
    '''
    The class for features which take continuous values
    
    Parameters
    ----------
    name : string
        the name of the feature
    min_value : float, default is None
        the minimal value for the feature
    max_value : float, default is None
        the maximal value for the feature
    mean_value : float, default is None
        the mean for the feature value distribution
    std_value : float, default is None
        the std for the feature value distribution 
    '''


    def __init__(self, name, min_value, max_value, mean_value, std_value):
        '''
        Constructor
        '''
        super().__init__(name)
        self.min_value = min_value
        self.max_value = max_value
        self.mean_value = mean_value
        self.std_value = std_value
        self.dtype = 'numeric'
        
    def to_dict(self):
        pdict = super().to_dict()
        cdict = {'dtype':self.dtype,'min_value':self.min_value,'max_value':self.max_value,'mean_value':self.mean_value,'std_value':self.std_value}
        return {**pdict,**cdict}
    
    
class CategoricFeature(BaseFeature):
    '''
    The class for features which take discrete values
    
    Parameters
    ----------
    name : string
        the name of the feature
    values : list
        the list of possible values for the feature
    
    '''


    def __init__(self, name, values):
        '''
        Constructor
        '''
        super().__init__(name)
        self.values = values
        self.dtype = 'categoric'
        
    def get_onehot_feature_names(self):
        """
        Get the one-hot encoding feature names for the possible values of the feature
        
        Returns
        -------
        list 
            the list of one-hot encoding feature names 
        """
        name_list = []
        for value in self.values:
            name_list.append(self.name+'='+str(value))
        return name_list
    
    def get_feature_name(self,value):
        """
        Get the one-hot encoding feature name for a possible value of the feature
        
        Parameters
        ----------
        value : object
            A possible value of the signal
        
        Returns
        -------
        string 
            the one-hot encoding feature name of the given value
        """
        return self.name+'='+str(value)
    
    def to_dict(self):
        pdict = super().to_dict()
        cdict = {'dtype':self.dtype,'values':self.values}
        return {**pdict,**cdict}
    
        