'''
Created on Aug 10, 2023

@author: Cheng Feng
'''

from . import helper

class UnivariateRule(object):
    '''
    A univariate PAR
    
    Parameters
    ----------
    feature : BaseFeature
        the feature
    '''
    def __init__(self, feature):
        '''
        Constructor
        '''
        self._feature = feature
        self._conf = 1
        self._support = 1
        
    def __str__(self):
        return helper.univariate_rule(self._feature)
    
    @property
    def conf(self):
        return self._conf
    
    @property
    def support(self):
        return self._support
    
    @property
    def feature(self):
        return self._feature

class Rule(object):
    '''
    A Predicate-based Association Rule
    
    Parameters
    ----------
    antec : list
        list of predicates in the antecedent set
    conseq : list
        list of predicates in the consequent set
    conf : float in [0,1]
        the confidence of the rule
    support : float in [0,1]
        the support of the rule
    '''
    def __init__(self, antec, conseq, conf, support):
        '''
        Constructor
        '''
        self._antec = antec
        self._conseq = conseq
        self._conf = conf
        self._support = support
    
    def __str__(self):
        strRule = ''
        for item in self._antec:
            strRule += item + ' and '
        strRule = strRule[0:len(strRule)-5] 
        strRule += ' ---> '
        for item in self._conseq:
            strRule += item + ' and '
        strRule = strRule[0:len(strRule)-5]
        return strRule
    
    @property
    def antec(self):
        return self._antec
    
    @property
    def conseq(self):
        return self._conseq
    
    @property
    def conf(self):
        return self._conf
    
    @property
    def support(self):
        return self._support
    
    def size(self):
        """
        get the number of predicates in the rule 
            
        Returns
        -------
        int 
            the number of predicates
        """
        return len(self._antec) + len(self._conseq)
    
    def pset(self):
        """
        get the set of predicates in the rule 
            
        Returns
        -------
        frozenset 
            the set of predicates
        """
        return frozenset( set(self._antec) | set(self._conseq) )
  
    def set_predicates(self,item_dict):
        antec = []
        for item in self._antec:
            antec.append(item_dict[item])
        self._antec = tuple(antec)
        
        conseq = []
        for item in self._conseq:
            conseq.append(item_dict[item])
        self._conseq = tuple(conseq)
    
    def extract_feats(self):
        """
        extract all the feature names from the rule 
            
        Returns
        -------
        list of strings
            list of feature names
        """
        antec_feats = []
        for item in self._antec:
            antec_feats.append( helper.extract_feat_from_predicate(item) )
        conseq_feats = []
        for item in self._conseq:
            conseq_feats.append( helper.extract_feat_from_predicate(item) )
        return conseq_feats+antec_feats
                