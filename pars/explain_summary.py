'''
Created on Aug 10, 2023

@author: Cheng Feng
'''


class ExplanationSummary(object):
    '''
    Explanation of reported anomaly
    '''
    def __init__(self):
        '''
        Constructor
        '''
        self._records = {}
        self._rule_feats_dict = {}
        self._rule_conf_dict = {}
        self._rule_sup_dict = {}
        
    
    def add_record(self, feat, location, score, rule, rule_conf, rule_sup, rule_feats):
        
        if self._records.get(feat, None) is None:
            self._records[feat] = [(location,score,rule,rule_feats)]
        else:
            self._records[feat].append( (location,score,rule,rule_feats) )
        
        self._rule_feats_dict[rule] = rule_feats
        self._rule_conf_dict[rule] = rule_conf
        self._rule_sup_dict[rule] = rule_sup
        
    
    def get_num_records(self):
        return len(self._records)
    
    def summary(self):
        """
        get explanation summary
            
        Returns
        -------
        list of tuples 
            [(anomalous feature,probability,violated rule,rule confidence,rule support,violated locations,related features),...]
        """
        feat_score_pairs = []
        sum_score = 0
        for feat in self._records.keys():
            # print('feat',feat)
            alarms = self._records[feat]
            rule_score = {}
            rule_location = {}
            score = 0
            for alarm in alarms:
                # print('alarm',alarm)
                score += alarm[1]
                if rule_score.get(alarm[2] , None) is None:
                    rule_score[alarm[2]] = alarm[1]
                    rule_location[alarm[2]] = [alarm[0]]
                else:
                    rule_score[alarm[2]] = rule_score[alarm[2]] + alarm[1]
                    rule_location[alarm[2]].append(alarm[0])
                # print('score',score)
                
            top_score = 0
            for rule in rule_score.keys():
                if rule_score[rule] > top_score:
                    top_score = rule_score[rule]
                    top_rule = rule
            
            sum_score += score
            feat_score_pairs.append( (feat,score,top_rule,self._rule_conf_dict[top_rule],self._rule_sup_dict[top_rule],rule_location[top_rule],self._rule_feats_dict[top_rule]) )
        
        feat_score_pairs = [(fsp[0],fsp[1]/sum_score,fsp[2],fsp[3],fsp[4],fsp[5],fsp[6]) for fsp in feat_score_pairs]
        
        sorted_feat_score_pairs = sorted(
            feat_score_pairs,
            key=lambda t: t[1],
            reverse=True
        )
        
        return sorted_feat_score_pairs
    
