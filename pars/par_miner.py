'''
Created on Aug 10, 2023

@author: z003w5we
'''
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from .rule import Rule
from .helper import filter_duplicate_rules
import logging

logger = logging.getLogger(__name__)


def _fpgrowth(data, max_len, min_sup, min_conf, index_dict):
    for entry in data:
        data.loc[data[entry]==1, entry] = index_dict[entry]
    df_list = data.values.tolist()
    dataset = []
    for datalist in df_list:
        temptlist = filter(lambda a: a != 0, datalist)
        numbers = list(temptlist)
        dataset.append(numbers)
            
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent = fpgrowth(df, min_support=min_sup,use_colnames=True,max_len=int(max_len))
      
    df = association_rules(frequent,metric='confidence',min_threshold=min_conf)
    rules = []
    for i in range(len(df)):
        antecedents = df.loc[i,'antecedents']
        consequents = df.loc[i,'consequents']
        confidence = df.loc[i,'confidence']
        support = df.loc[i,'support']
        rule = Rule(antecedents,consequents,confidence,support)
        rules.append(rule)
    return rules


def _multi_fpgrowth(df, max_len, min_sup, min_conf,index_dict,max_perdicts_for_rule_mining, max_times_for_rule_mining):
    rules_all = []
    for i in range(max_times_for_rule_mining):
        logger.debug(f'Start rule ming process:{i+1}/{max_times_for_rule_mining}')
        data = df.sample(n=max_perdicts_for_rule_mining,axis='columns')
        rules = _fpgrowth(data,max_len,min_sup,min_conf,index_dict)
        rules_all.extend(rules)
    return rules_all

def mine_rules(df, max_len, min_sup, min_conf, max_predicts_for_rule_mining, max_times_for_rule_mining):
    index_dict = {}
    item_dict = {}
    index = 100
    for entry in df:
        index_dict[entry] = index
        item_dict[index] = entry
        index += 1
   
    if df.shape[1] <= max_predicts_for_rule_mining or max_times_for_rule_mining<=1:
        data = df.copy()
        rules= _fpgrowth(data, max_len, min_sup, min_conf, index_dict)
    else:
        rules = _multi_fpgrowth(df, max_len, min_sup,min_conf,index_dict,max_predicts_for_rule_mining, max_times_for_rule_mining)
    
    for rule in rules:
        rule.set_predicates(item_dict)
    rules = filter_duplicate_rules(rules)
    return rules,set(item_dict.values())