'''
Created on Aug 11, 2023

@author: z003w5we
'''

import pandas as pd
import zipfile
from pars import NumericFeature, CategoricFeature, PARAnomalyExplainer

if __name__ == '__main__':
    z_tr = zipfile.ZipFile('../data/SWaT_train.zip', "r")
    f_tr = z_tr.open(z_tr.namelist()[0])
    train_df=pd.read_csv(f_tr)
    f_tr.close()
    z_tr.close()
    
    z_tr = zipfile.ZipFile('../data/SWaT_test.zip', "r")
    f_tr = z_tr.open(z_tr.namelist()[0])
    test_df=pd.read_csv(f_tr)
    f_tr.close()
    z_tr.close()
    
    train_df = train_df.drop(['Timestamp', 'Normal/Attack'], axis=1)
    
    features = []
    for name in train_df.columns:
        if len(train_df[name].unique()) > 5:
            features.append( NumericFeature(name,min_value=train_df[name].min(), max_value=train_df[name].max(),
                                        mean_value=train_df[name].mean(), std_value=train_df[name].std()) )
        else:
            features.append( CategoricFeature(name,values=train_df[name].unique().tolist()) )
    
    parexp = PARAnomalyExplainer(features)
    parexp.train(train_df, max_predicts4rule_mining=50, max_times4rule_mining=2)
    
    feature_names = [feature.name for feature in features]
    test_df = test_df.loc[test_df['label']==1,:].reset_index(drop=True)
    
    rules = parexp.find_violated_pars(test_df.loc[0,feature_names].values, topk=5)
    for rule in rules:
        print(rule)
        
    
    explanation = parexp.explain_anomalies(test_df.loc[0:10,feature_names].values, labels=test_df.loc[0:10,'label'].values, topk=5)
    for exp_item in explanation.summary():
        print(exp_item)