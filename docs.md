# Documentation
---
### CategoricalClassification.generate_data
```python
CategoricalClassification.generate_data(n_features, n_samples, cardinality=5, ensure_rep=False, seed=42)
```
Generates dataset of shape (n_samples, n_features), based on given parameters.

- **n_features:** _int_
  The number of features in a generated dataset.
- **n_samples:** _int_
  The number of samples in a generated dataset.
- **cardinality:** _int, list, numpy.ndarray_, default=5.
  Sets the cardinality and shape of a generated dataset.
  -  If int, the entire dataset has a cardinality of given int, with roughly normal density distribution of features, with a randomly selected feature value as a peak.
  -  
