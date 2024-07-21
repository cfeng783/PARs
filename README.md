# PARs: Predicate-based Association Rules for Efficient and Accurate Model-Agnostic Anomaly Explanation

This repository includes the Python package which offers functionalities for explaining data anomalies detected by arbitrary models using PARs. The methodology of PARs is described in the follow paper:  

Cheng Feng. 2024. PARs: Predicate-based Association Rules for Efficient and Accurate Anomaly Explanation. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (CIKM ’24), October 21–25, 2024, Boise, ID, USA. ACM, New York, NY, USA, 10 pages. https: //doi.org/10.1145/3627673.3679625.

An extended version is available at https://arxiv.org/abs/2312.10968.

The scripts for experiments in the paper is available at https://github.com/cfeng783/PARs-Exp.

#### Install dependencies

```shell
pip install -r requirements.txt
```

#### How to use the PARs package

```python
### define features and the PARAnomalyExplainer
from pars import NumericFeature, CategoricFeature, PARAnomalyExplainer

features = []
for name in train_df.columns:
    if len(train_df[name].unique()) > 5:
        features.append( NumericFeature(name,min_value=train_df[name].min(), max_value=train_df[name].max(),
                                    mean_value=train_df[name].mean(), std_value=train_df[name].std()) )
    else:
        features.append( CategoricFeature(name,values=train_df[name].unique().tolist()) )

parexp = PARAnomalyExplainer(features)

### let's train the PARAnomalyExplainer
parexp.train(train_df, max_predicts4rule_mining = 75, max_times4rule_mining = 5, set_seed=False)

### you can use PARAnomalyExplainer to find top-k violated PARs for an individual anomaly
rules = parexp.find_violated_pars(anomalies[0], topk=5)
print('Violated PARs:')
for rule in rules:
    print(f'{rule}, sup: {rule.support}, conf: {rule.conf}')

### you can also find summarized anomaly explanation for a list of anomalies
explanation = parexp.explain_anomalies(anomalies[0:20])
# each explanation item is a tuple contains the following elements: 
# (anomalous feature,probability,violated rule,rule confidence,rule support,violated locations,related features)
for exp_item in explanation.summary():
    print(f'anomalous feature: {exp_item[0]}')
    print(f'probability: {exp_item[1]}')
    print(f'representive violated PAR: {exp_item[2]}')
    print(f'confidence of representive PAR: {exp_item[3]}')
    print(f'support of the representive PAR: {exp_item[4]}')
    print(f'violated locations: {exp_item[5]}')
    print(f'related features in the representive PAR: {exp_item[6]}')
    print()

```

Please check [tutorial.ipynb](tutorial.ipynb) for the complete tutorial.
