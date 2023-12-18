'''
Created on Aug 10, 2023

@author: Cheng Feng
'''

import random,pickle,os
from .par_miner import mine_rules
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from .features import NumericFeature
from .explain_summary import ExplanationSummary
from .utils import DataUtil
import numpy as np
from . import helper
import logging
import pandas as pd
from .rule import UnivariateRule

logger = logging.getLogger(__name__)

class PARAnomalyExplainer:
    '''
    PARs for model-agnostic anomaly explanation
    
    Parameters
    ----------
    features : list of Features
        the features, each feature is either a NumericFeature or CategoricFeature
    wsup : float
        importance weight for support factor  
    wconf : float
       importance weight for confidence factor    
    minsup : float in [0,1]
        lower bound for the support of a predicate  
    minconf : float in [0,1]
        lower bound for the confidence of a generated rule
    max_predicates_per_rule : int
        max number of predicates in a generated rule 
    maxsup : float in [0,1]
        upper bound for the support of a predicate, used only for efficiency purpose
    '''
    def __init__(self, features=None, wsup=1, wconf=5, minsup=0.01, minconf=0.95, max_predicates_per_rule=5, maxsup=0.99):
        if features is not None: ##indicate not loading from file system
            self._du = DataUtil(features)
            self._init_features()
            
            self._wsup = wsup
            self._wconf = wconf
            self._minsup = minsup
            self._minconf = minconf
            self._max_predicates_per_rule = max_predicates_per_rule
            self._maxsup = maxsup
            
    def _init_features(self):
        self._features = self._du.features
        self._cont_featnames = []
        self._disc_featnames = []
        self._numeric_features = []
        self._categoric_features = []
        for feature in self._features:
            if isinstance(feature, NumericFeature):
                self._cont_featnames.append(feature.name)
                self._numeric_features.append(feature)
            else:
                self._disc_featnames.extend(feature.get_onehot_feature_names())
                self._categoric_features.append(feature)
        
    def _propose_cutoff_values(self,df,min_samples_leaf):
        cutoffs = {}
        for feature in self._numeric_features:
            cutoffs[feature.name] = []
        
        for feature in self._categoric_features:
            if len(self._cont_featnames) > 0:
                onehot_feats = feature.get_onehot_feature_names()
                if len(onehot_feats) <= 1:
                    continue
                
                df.loc[:,'tempt_label'] = 0
                for i in range(len(onehot_feats)):
                    df.loc[df[onehot_feats[i]]==1,'tempt_label'] = i

                xfeats = list(self._cont_featnames)
                x = df[xfeats].values
                y = df['tempt_label'].values
                df.drop(columns='tempt_label',inplace=True)
                model = DecisionTreeClassifier(criterion = "entropy",min_samples_leaf=int(min_samples_leaf))
                model.fit(x,y)
                cut_tuples = helper.extract_cutoffs(model.tree_,xfeats)
                for ct in cut_tuples:
                    cutoffs[ct[0]].append( (ct[1],ct[2]) )
        
        for feature in self._numeric_features:
            yfeat = feature.name
            xfeats = list(self._cont_featnames)
            xfeats.remove(yfeat)
            
            if len(xfeats) == 0:
                continue
            
            x = df[xfeats].values
            y = df[yfeat].values
            y = (y-feature.mean_value)/feature.std_value
            model = DecisionTreeRegressor(min_samples_leaf=int(min_samples_leaf))
            model.fit(x,y)
            cut_tuples = helper.extract_cutoffs(model.tree_,xfeats)
            for ct in cut_tuples:
                cutoffs[ct[0]].append( (ct[1],ct[2]) )
    
        cutoffs = helper.reset_cutoffs(cutoffs,df,min_samples_leaf)
        return cutoffs
        
    def train(self, data, max_predicts4rule_mining = 100, max_times4rule_mining = 5, set_seed=False):
        """
        learn PARs from data
        
        Parameters
        ----------
        data : DataFrame or ndarray
            the training data
        max_predicts4rule_mining : int
            the max number of predicts can be allowed in a rule mining process (in order to speed up mining process)
        max_times4rule_mining : int
            the max number for the rule mining process, only use for when number of generated predicates is larger than max_perdicts_for_rule_mining
        set_seed : Bool
            whether set random seed for reproducibility
        
        Returns
        ------------
        PARExplainer
            self
        
        """
        if isinstance(data, np.ndarray):
            df = pd.DataFrame(data=data, columns=[feature.name for feature in self._features])
            df = self._du.onehot_encode(df)
        elif isinstance(data, pd.DataFrame):
            df = self._du.onehot_encode(data)
        else:
            logger.error('unsupported data type!')
            return self
            
        if set_seed:
            np.random.seed(123)
            random.seed(1234)
        
        min_samples_predicate = int(self._minsup * len(df))
        

        cutoffs = self._propose_cutoff_values(df, min_samples_leaf=min_samples_predicate)
        for feat in self._cont_featnames:
            if feat not in cutoffs:
                df.drop(columns=feat,inplace=True)
                continue
            
            vals2preserve = [val_pri_pair[0] for val_pri_pair in cutoffs[feat]]
            vals2preserve.sort()
                      
            for j in range(len(vals2preserve)):
                if j == 0:
                    new_feat = feat + '<' + str(vals2preserve[j])
                    df[new_feat] = 0
                    df.loc[df[feat] < vals2preserve[j], new_feat] = 1
                if j > 0:
                    new_feat = str(vals2preserve[j-1]) + '<=' + feat + '<' + str(vals2preserve[j])
                    df[new_feat] = 0
                    df.loc[(df[feat]<vals2preserve[j]) & (df[feat]>=vals2preserve[j-1]), new_feat] = 1
                if j == len(vals2preserve)-1:
                    new_feat = feat + '>=' + str(vals2preserve[j])
                    df[new_feat] = 0
                    df.loc[df[feat] >= vals2preserve[j], new_feat] = 1
            df.drop(columns=feat,inplace=True)

        low_feats = []
        for entry in df.columns:
            support = len(df.loc[df[entry]==1,:])
            if support < min_samples_predicate and entry.find('<') == -1 and entry.find('>') == -1:
                low_feats.append( entry )
        
        start_index = 0
        combine_list = []
        for i in range(1,len(low_feats)):
            df.loc[:,'tempt'] = 0
            for j in range(start_index,i):
                df.loc[df[low_feats[j]]==1, 'tempt'] = 1
            left_support = len(df.loc[df['tempt']==1,:])
             
            if left_support >= min_samples_predicate:
                df.loc[:,'tempt'] = 0
                for j in range(i+1,len(low_feats)):
                    df.loc[df[low_feats[j]]==1, 'tempt'] = 1
                right_support = len(df.loc[df['tempt']==1,:])
                if right_support >= min_samples_predicate:
                    combine_list.append(low_feats[start_index:i+1])
                    start_index = i+1
                else:
                    combine_list.append(low_feats[start_index:])
                    break
        df = df.drop(columns='tempt',errors='ignore')
        
        for entry2combine in combine_list:
            new_feat = entry2combine[0]
            for i in range(1,len(entry2combine)):
                new_feat += ' or ' + entry2combine[i]
            df[new_feat] = 0
            for feat in entry2combine:
                df.loc[df[feat]==1, new_feat] = 1
                df = df.drop(columns=feat)
        
        for entry in df.columns:
            support = len(df.loc[df[entry]==1,:])/len(df)
            if support > self._maxsup or support < self._minsup:
                df = df.drop(columns=entry)
        
        ## start invariant rule mining
        rules, predicates = mine_rules(df, max_len=self._max_predicates_per_rule,
                                        min_sup=self._minsup,
                                        min_conf = self._minconf,
                                        max_predicts_for_rule_mining = max_predicts4rule_mining,
                                        max_times_for_rule_mining = max_times4rule_mining)
        self._predicates = predicates
        self._rules = sorted(rules, key=lambda t: self._wsup*(t.support-self._minsup)/(1-self._minsup) + self._wconf*(t.conf-self._minconf)/(1-self._minconf), reverse=True)
        
        logger.info(f'number of predicates: {len(self._predicates)}')
        logger.info(f'number of rules: {len(self._rules)+len(self._features)}')
        
        return self
    
    
    def explain_anomalies(self,data,topk=5,labels=None):
        """
        get the explanation for anomalies in given data
        
        Parameters
        ----------
        data : DataFrame or ndarray
            the data
        labels : list of bool, default is None
            the predicted labels for data, 1: anomaly, 0: normal. Note len(data) should equal to len(labels).
            If all the data are anomaly, set labels to None
        topk : int
            only topk PARs are selected for each anomaly
            
        Returns
        -------
        ExplanationSummary 
            explanation summary
        """
        if isinstance(data, np.ndarray):
            df = pd.DataFrame(data=data, columns=[feature.name for feature in self._features])
            df = self._du.onehot_encode(df)
        elif isinstance(data, pd.DataFrame):
            df = self._du.onehot_encode(data)
            df = df.reset_index(drop=True)
        else:
            logger.error('unsupported data type!')
            return None
        
        if labels is None:
            labels = [1]*len(df)
        
        feature_std = {}
        for feature in self._numeric_features:
            feature_std[feature.name] = feature.std_value
        df = helper.parse_predicates(df,self._predicates,feature_std)
        
        exp = ExplanationSummary()
        
        for feature in self._features:
            scores = df.apply(helper.boundary_anomaly_scoring,axis=1,args=(feature,feature_std)).to_numpy()
            indices = np.where(scores>0)[0]
            for i in indices:
                exp.add_record(feature.name, i, self._wsup+self._wconf, helper.univariate_rule(feature), 1, 1, [feature.name])
        
        for i in range(len(df)):
            if labels[i] == 0:
                continue
            # print(i,'/',len(test_df))
            k = 0
            for rule in self._rules:
                antec_satisfy = True
                
                for item in rule.antec:
                    if df.loc[i,item] != 1:
                        antec_satisfy = False
                        break
                
                if antec_satisfy:
                    for item in rule.conseq:
                        if df.loc[i,item] != 1:
                            feat = helper.extract_feat_from_predicate(item)
                            score = self._wsup*(rule.support-self._minsup)/(1-self._minsup) + self._wconf*(rule.conf-self._minconf)/(1-self._minconf)
                            if feat is not None:
                                exp.add_record(feat, i, score, str(rule), rule.conf, rule.support, rule.extract_feats())
                                k += 1
                if k >= topk:
                    break
        return exp
    
    def find_violated_pars(self,anomaly,topk=5):
        """
        find the top-k violated PARs for an individual anomaly
        
        Parameters
        ----------
        anomaly : ndarray
            the anomaly data, shape=(1,len(features)) or len(features)
        topk : int
            only topk PARs are selected for each anomaly
            
        Returns
        -------
        list of Rule or UnivariateRule
            the top-k violated PARs
        """
        
        data = np.reshape(anomaly,(1,-1))
        df = pd.DataFrame(data=data, columns=[feature.name for feature in self._features])
        test_df = self._du.onehot_encode(df)
        feature_std = {}
        for feature in self._numeric_features:
            feature_std[feature.name] = feature.std_value
        test_df = helper.parse_predicates(test_df,self._predicates,feature_std)
         
        topk_rules = []
        
        scores = []
        for feature in self._features:
            score = test_df.apply(helper.boundary_anomaly_scoring,axis=1,args=(feature,feature_std)).to_numpy()[0]
            if score > 0:
                topk_rules.append( UnivariateRule(feature) )
                scores.append(score)
        
        temp = sorted(scores,reverse=True)
        if len(topk_rules) >= topk:
            res = []
            for i in range(topk):
                res.append(topk_rules[scores.index(temp[i])])
            return res
                   
        for rule in self._rules:
            antec_satisfy = True
            conseq_satisfy = True
            
            for item in rule.antec:
                if test_df.loc[0,item] != 1:
                    antec_satisfy = False
                    break
            
            if antec_satisfy:
                for item in rule.conseq:
                    if test_df.loc[0,item] != 1:
                        conseq_satisfy = False
                        break
                
                if not conseq_satisfy:
                    topk_rules.append( rule )
                    if len(topk_rules) >= topk:
                        break
        return topk_rules
              
    def export_rules(self, filepath):
        """
        export rules to file
        
        Parameters
        ----------
        filepath : string
            the file path
        """
        with open(filepath,'w') as myfile:
            for feature in self._features:
                myfile.write(helper.univariate_rule(feature) + '\n')
            for rule in self._rules:
                myfile.write(str(rule) + '\n')
            myfile.close()
    
    def get_num_rules(self):
        """
        get number of rules
        
        Returns
        -------
        int
            the number of rules
        """
        return len(self._rules)+len(self._features)
    
    def save_model(self,model_path):
        """
        save the model to files
        
        Parameters
        ----------
        model_path : string, default is None
            the target folder where the model files are saved.
        """
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        pickle.dump(self._rules,open( os.path.join(model_path,'rules.pkl'),'wb'))
        pickle.dump(self._predicates,open( os.path.join(model_path,'predicates.pkl'),'wb'))
        config_dict = dict(
            wsup = self._wsup,
            wconf = self._wconf, 
            minsup = self._minsup, 
            minconf = self._minconf,
            max_predicates_per_rule = self._max_predicates_per_rule,
            maxsup = self._maxsup
            )
        pickle.dump(config_dict,open(os.path.join(model_path,'config_dict.pkl'),'wb'))
        self._du.save2jsonfile(os.path.join(model_path,'data_util.json'))
        # self.export_rules(os.path.join(model_path,'rules.txt'))
         
    def load_model(self,model_path):
        """
        load the model from files
        
        Parameters
        ----------
        model_path : string, default is None
            the target folder where the model files are located
            
        Returns
        -------
        AnomalyExplainer
            self
        """
        self._predicates = pickle.load(open(os.path.join(model_path,'predicates.pkl'),'rb'))
        self._rules = pickle.load(open(os.path.join(model_path,'rules.pkl'),'rb'))
        config_dict = pickle.load(open(os.path.join(model_path,'config_dict.pkl'),'rb'))
        self._wsup = config_dict['wsup']
        self._wconf = config_dict['wconf']
        self._minsup = config_dict['minsup']
        self._minconf = config_dict['minconf']
        self._max_predicates_per_rule = config_dict['max_predicates_per_rule']
        self._maxsup = config_dict['maxsup']
        
        self._du = DataUtil(filename=os.path.join(model_path,'data_util.json'))
        self._init_features()
        return self