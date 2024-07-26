# Categorical Classification
M. Sc. Project - Artifical Categorical Datasets

[![selftest](https://github.com/98MM/msc_cc/actions/workflows/test_cc.yml/badge.svg)](https://github.com/98MM/msc_cc/actions/workflows/test_cc.yml)

This project is also hosted on [Outrank](https://github.com/outbrain/outrank).

- Source code in src/generator.py
- Demo in src/CC Demo.ipnyb
---
# Usage
```bash
pip install catclass
```
### Creating a simple dataset
```python
from catclass import Categorical Classification

cc = CategoricalClassification()

# Creates a simple dataset of 10 features, 10k samples, with feature cardinality of all features being 35
X = cc.generate_data(10, 
                     10000, 
                     cardinality=35, 
                     ensure_rep=True, 
                     random_values=True, 
                     low=0, 
                     high=40)

# Creates target labels via clustering
y = cc.generate_labels(X, n=2, class_relation='cluster')
