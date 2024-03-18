# Documentation
---
### CategoricalClassification.generate_data
```python
CategoricalClassification.generate_data(n_features, n_samples, cardinality=5, ensure_rep=False, seed=42)
```
Generates dataset of shape (n_samples, n_features), based on given parameters.

- **n\_features:** _int_
  The number of features in a generated dataset.
- **n\_samples:** _int_
  The number of samples in a generated dataset.
- **cardinality:** _int, list, numpy.ndarray_, default=5.
  Sets the cardinality and shape of a generated dataset.
  -  If _int_, the entire dataset has a cardinality of given int, with roughly normal density distribution of values, with a randomly selected value as a peak. The feature values will be integers, in range \[0, cardinality\].
  -  _list_ or _numpy.ndarray_ offers more control and follows the format \[_int_, _tuple_, _tuple_, ...\], where:
    - _int_ represents the default cardinality of a generated dataset
    - _tuple_ can either be:
      - (_int_ or _list_, _int_): the first element represents the index or list of indexes of features. The second element their cardinality. Generated features will have a roughly normal density distribution of values, with a randomly selected value as a peak. The feature values will be integers, in range \[0, second element of tuple\].
      - (_int_ or _list_, _list_): the first element represents the index or list of indexes of features. The second element offers two options:
        - _list_:  a list of values,
        - \[_list_, _list_\]: where the first _list_ element represents a set of values the feature or features posses, the second the frequencies or probabilities of individual features.
- **seed**: _int_, default=42.
  Controls _numpy.random.seed_               

**Returns**: a _numpy.ndarray_ dataset with **n\_features** features and **n\_samples** samples.
