'''
Created on Aug 10, 2023

@author: Cheng Feng
'''
import numpy as np
import logging
from .features import NumericFeature

logger = logging.getLogger(__name__)

def parse_predicates(df, predicates, sigstd):
    for item in predicates:
        scores = df.apply(fuzzy_membership,axis=1,args=(item,sigstd,))
        df[item] = scores
    return df

def fuzzy_membership(x,predicate,sigstd):
    score = 0
    if predicate.find(' or ') != -1:
        subitems = predicate.split(' or ')
        satisfy = False
        for subitem in subitems:
            if x[subitem]==1:
                satisfy = True
                break
        if satisfy:
            score = 1
        else:
            score = 0
    else:
        if predicate.find('<=') != -1:
            pos1, pos2 = predicate.find('<='), predicate.rfind('<')
            lb = float(predicate[:pos1])
            ub = float(predicate[pos2+1:])
            feat = predicate[pos1+2:pos2]
            if x[feat] >= lb and x[feat] <= ub:
                score = 1
            elif x[feat] < lb:
                score = 1-max(1, (lb-x[feat])/(sigstd[feat]+1e-5) )
            elif x[feat] > ub:
                score = 1-max(1, (x[feat]-ub)/(sigstd[feat]+1e-5) )
        elif predicate.find('<') != -1:
            pos = predicate.find('<')
            ub = float(predicate[pos+1:])
            feat = predicate[:pos]
            if x[feat] <= ub:
                score = 1
            else:
                score = 1-max(1, (x[feat]-ub)/(sigstd[feat]+1e-5) )
        elif predicate.find('>=') != -1:
            pos = predicate.find('>=')
            lb =  float(predicate[pos+2:])
            feat = predicate[:pos]
            if x[feat] >= lb:
                score = 1
            else:
                score = 1-max(1, (lb-x[feat])/(sigstd[feat]+1e-5) )
        else:
            if x[predicate] == 1:
                score = 1
            else:
                score = 0        
    return score

def boundary_anomaly_scoring(x,feature,sigstd):    
    if isinstance(feature, NumericFeature):
        lb = min(feature.min_value, feature.mean_value-3*feature.std_value)
        ub = max(feature.max_value, feature.mean_value+3*feature.std_value)
        feat = feature.name
        # print(feat,lb,ub)
        if x[feat] >= lb and x[feat] <= ub:
            score = 0
        elif x[feat] < lb:
            score = max(1, (lb-x[feat])/(sigstd[feat]+1e-8) )
        elif x[feat] > ub:
            score = max(1, (x[feat]-ub)/(sigstd[feat]+1e-8) )
    else:
        satisfy = False
        for entry in feature.get_onehot_feature_names():
            if x[entry] == 1:
                satisfy = True
                break
        if satisfy:
            score = 0
        else:
            score = 1
    return score

def extract_feat_from_predicate(item):
    if item.find(' or ') != -1:
        return None
    else:
        if item.find('<=') != -1:
            pos1, pos2 = item.find('<='), item.rfind('<')
            feat = item[pos1+2:pos2]
        elif item.find('<') != -1:
            pos = item.find('<')
            feat = item[:pos]
        elif item.find('>=') != -1:
            pos = item.find('>=')
            feat = item[:pos]
        else:
            pos = item.rfind('=')
            feat = item[:pos]
    return feat

def filter_duplicate_rules(rules):
    logger.debug(f'number of rules before filtering: {len(rules)}')
    rule_str_set = set()
    rules2keep = []
    for rule in rules:
        if len(rule.conseq)==1 and str(rule) not in rule_str_set:
            rules2keep.append(rule)
            rule_str_set.add(str(rule))
    rules2keep = sorted(rules2keep, key=lambda t: t.size())
    logger.debug(f'number of rules after filtering: {len(rules2keep)}')
    final_rules = []
    sup_rule_dict = {}
    for rule in rules2keep: 
        candidate_rules = sup_rule_dict.get(rule.support, [])
        dup = False
        for exrule in candidate_rules:
            if set(exrule.conseq) == set(rule.conseq) and set(exrule.antec).issubset( set(rule.antec) ):
                dup = True
                break
        if not dup:
            final_rules.append(rule)
            candidate_rules.append(rule)
            sup_rule_dict[rule.support] = candidate_rules
    logger.debug(f'number of final rules: {len(final_rules)}')
    return final_rules


def univariate_rule(feature):
    if isinstance(feature, NumericFeature):
        lb = min(feature.min_value, feature.mean_value-3*feature.std_value)
        ub = max(feature.max_value, feature.mean_value+3*feature.std_value)
        strOut = str(lb)+'<='+feature.name+'<='+str(ub)            
    else:
        strOut = feature.name + ' in ' + str(feature.get_onehot_feature_names())
    return strOut

def search_insert_position(vals, val2insert):
    pos = None
    for i in range(len(vals)):
        if val2insert <= vals[i]:
            pos = i
    
    if pos is None:
        pos = len(vals)
    return pos

def reset_cutoffs(cutoffs,df,min_samples):
    preserved_cutoffs = {}
    for feat in cutoffs.keys():
        if len(cutoffs[feat]) == 0:
            preserved_cutoffs[feat] = []
            continue
        
        value_priority_pairs = cutoffs[feat]
        sorted_value_priority_pairs = sorted(
            value_priority_pairs,
            key=lambda t: [t[1],-t[0]],
            reverse=True
        )
        # print(feat,sorted_value_priority_pairs)
        for i in range(len(sorted_value_priority_pairs)):
            init_val = sorted_value_priority_pairs[i][0]
            if len( df.loc[ df[feat]<init_val,:] ) >= min_samples and len( df.loc[ df[feat]>init_val,:] ) >= min_samples:
                sorted_value_priority_pairs = sorted_value_priority_pairs[i:]
                break
        
        
        vals = [pair[0] for pair in sorted_value_priority_pairs]
        pair2preserve = [sorted_value_priority_pairs[0]]
        val2preserve = []
        val2preserve.append(vals[0])
        for i in range(1, len(vals)):
            pos = search_insert_position(val2preserve, vals[i])
            if pos == 0:
                if len( df.loc[ (df[feat]>=vals[i]) & (df[feat]<val2preserve[0]),:] ) >= min_samples and \
                    len( df.loc[ df[feat]<vals[i],:] ) >= min_samples:
                    val2preserve.insert(0,vals[i])
                    pair2preserve.append(sorted_value_priority_pairs[i])
            elif pos == len(val2preserve):
                if len( df.loc[ (df[feat]<vals[i]) & (df[feat]>=val2preserve[-1]),:] ) >= min_samples and \
                    len( df.loc[ df[feat]>=vals[i],:] ) >= min_samples:
                    val2preserve.append(vals[i])
                    pair2preserve.append(sorted_value_priority_pairs[i])
            else:
                if len( df.loc[ (df[feat]<vals[i]) & (df[feat]>=val2preserve[pos-1]),:] ) >= min_samples and \
                    len(df.loc[(df[feat] < val2preserve[pos]) & (df[feat] >= vals[i]),:]) >= min_samples:
                    val2preserve.insert(pos, vals[i])
                    pair2preserve.append(sorted_value_priority_pairs[i])
        preserved_cutoffs[feat] = pair2preserve
    return preserved_cutoffs


def extract_cutoffs(dtree,feats):
    n_nodes = dtree.node_count
    children_left = dtree.children_left
    children_right = dtree.children_right
    feature = dtree.feature
    # print("feature",feature)
    threshold = dtree.threshold
    node_samples = dtree.n_node_samples
    impurity = dtree.impurity
    N = dtree.n_node_samples[0]
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    cutoffs = []
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
            
    for i in range(n_nodes):
        if is_leaves[i] == False:
            c_l, c_r = children_left[i], children_right[i]
            N_t, N_t_L, N_t_R = node_samples[i], node_samples[c_l], node_samples[c_r]
            imp, imp_l, imp_r = impurity[i], impurity[c_l], impurity[c_r]
            impurity_decrease = N_t / N * (imp - N_t_R/N_t*imp_r - N_t_L/N_t*imp_l)

            if type(feats) == list:
                cutoffs.append( (feats[feature[i]],threshold[i], impurity_decrease) )
            else:
                cutoffs.append( (feats,threshold[i], impurity_decrease) )
    return cutoffs