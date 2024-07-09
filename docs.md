# Documentation
---

### CategoricalClassification.dataset_info
```python
print(CategoricalClassification.dataset_info)
```
Stores a formatted dictionary of operations made. Function _CategoricalClassification.generate\_data_ resets its contents. Each subsequent function call adds information to it.

---

### CategoricalClassification.generate_data
```python
CategoricalClassification.generate_data(n_features, n_samples, cardinality=5, structure=None, ensure_rep=False, random_values=False, low=0, high=1000, seed=42)
```
Generates dataset of shape **_(n_samples, n_features)_**, based on given parameters.

- **n\_features:** _int_
  The number of features in a generated dataset.
- **n\_samples:** _int_
  The number of samples in a generated dataset.
- **cardinality:** _int_, default=5.
  Sets the default cardinality of a generated dataset.
- **structure:** _list, numpy.ndarray_, default=None.
  Sets the structure of a generated dataset. Offers more controle over feature value domains and value distributions.
  Follows the format **\[_tuple_, _tuple_, ...\]**, where:
   - **_tuple_** can either be:
      - **(_int_ or _list_, _int_)**: the first element represents the index or list of indexes of features. The second element their cardinality. Generated features will have a roughly normal density distribution of values, with a randomly selected value as a peak. The feature values will be integers, in range \[0, second element of tuple\].
      - **(_int_ or _list_, _list_)**: the first element represents the index or list of indexes of features. The second element offers two options:
        - **_list_**:  a list of values to be used in the feature or features,
        - **\[_list_, _list_\]**: where the first _list_ element represents a set of values the feature or features posses, the second the frequencies or probabilities of individual features.
- **ensure_rep:** _bool_, default=False:
  Control flag. If **_True_**, all possible values **will** appear in the feature.
- **random_values:** _bool_, default=False:
  Control flag. If **_True_**, value domain of feature will be random on interval _\[low, high\]_.
- **low**: _int_
  Sets lower bound of value domain of feature.
- **high**: _int_
  Sets upper bound of value domain of feature. Only used when _random\_values_ is True.
- **seed**: _int_, default=42.
  Controls **_numpy.random.seed_**               

**Returns**: a **_numpy.ndarray_** dataset with **n\_features** features and **n\_samples** samples.

---
### CategoricalClassification.\_feature\_builder
```python
CategoricalClassification._feature_builder(feature_attributes, n_samples, ensure_rep=False, random_values=False, low=0, high=1000)
```
Helper function used to avoid duplicate code blocks in _generate\_data_. Generates feature array based on _feature\_attributes._

- **feature\_attributes**: _int_ or _list_ or _numpy.ndarray_
Attributes of feature. Can be just cardinality (_int_), value domain (_list_), or value domain and their respective probabilities  (_list_).
- **n\_samples**: _int_
Number of samples in dataset. Determines generated feature vector size.
- **ensure_rep:** _bool_, default=False:
  Control flag. If **_True_**, all possible values **will** appear in the feature.
- **random_values:** _bool_, default=False:
  Control flag. If **_True_**, value domain of feature will be random on interval _\[low, high\]_.
- **low**: _int_
  Sets lower bound of value domain of feature.
- **high**: _int_
  Sets upper bound of value domain of feature. Only used when _random\_values_ is True.

**Returns:** a **_numpy.ndarray_** feature array.

---

### CategoricalClassification.\_generate\_feature
```python
CategoricalClassification._generate_feature(size, vec=None, cardinality=5, ensure_rep=False, random_values=False, low=0, high=1000 p=None)
```
Generates feature array of length **_size_**. Called by _CategoricalClassification.generate\_data_, by utilizing _numpy.random.choice_. If no probabilites array is given, the value density of the generated feature array will be roughly normal, with a randomly chosen peak. The peak will be chosen from the value array.

- **size**: _int_
  Length of generated feature array.
- **vec**: _list_ or _numpy.ndarray_, default=None
  List of feature values, value domain of feature.
- **cardinality**: _int_, default=5
  Cardinality of feature to use when generating its value domain. If _vec_ is not None, vec is used instead.
- **ensure_rep**: _bool_, default=False
  Control flag. If **_True_**, all possible values **will** appear in the feature array.
- **random_values:** _bool_, default=False:
  Control flag. If **_True_**, value domain of feature will be random on interval _\[low, high\]_.
- **low**: _int_
  Sets lower bound of value domain of feature.
- **high**: _int_
  Sets upper bound of value domain of feature. Only used when _random\_values_ is True.
- **p**: _list_ or _numpy.ndarray_, default=None
  Array of frequencies or probabilities. Must be of length _v_ or equal to the length of _v_.

**Returns:** a **_numpy.ndarray_** feature array. 

___

### CategoricalClassification.generate\_combinations
```python
CategoricalClassification.generate_combinations(X, feature_indices, combination_function=None, combination_type='linear')
```
Generates and adds a new column to given dataset **X**. The column is the result of a combination of features selected with **feature\_indices**. Combinations can be linear, nonlinear, or custom defined functions.

- **X**: _list_ or _numpy.ndarray_:
  Dataset to perform the combinations on.
- **feature_indices**: _list_ or _numpy.ndarray_:
  List of feature (column) indices to be combined.
- **combination\_function**: _function_, default=None:
  Custom or user-defined combination function. The function parameter **must** be a _list_ or _numpy.ndarray_ of features to be combined. The function **must** return a _list_ or _numpy.ndarray_ column or columns, to be added to given dataset _X_ using _numpy.column\_stack_.
- **combination\_type**: _str_ either _linear_ or _nonlinear_, default='linear':
  Selects which built-in combination type is used.
  - If _'linear'_, the combination is a sum of selected features.
  - If _'nonlinear'_, the combination is the sine value of the sum of selected features.

**Returns:** a **_numpy.ndarray_** dataset X with added feature combinations.

---

### CategoricalClassification.\_xor
```python
CategoricalClassification._xor(arr)
```
Performs bitwise XOR on given vectors and returns result.
- **arr**: _list_ or _numpy.ndarray_
  List of features to perform the combination on.

**Returns:** a **_numpy.ndarray_** result of **_numpy.bitwise\_xor(a,b)_** on given columns in **_arr_**.

___

### CategoricalClassification.\_and
```python
CategoricalClassification._and(arr)
```
Performs bitwise AND on given vectors and returns result.
- **arr**: _list_ or _numpy.ndarray_
  List of features to perform the combination on.

**Returns:** a **_numpy.ndarray_** result of **_numpy.bitwise\_and(a,b)_** on given columns in **_arr_**.


___

### CategoricalClassification.\_or
```python
CategoricalClassification._or(arr)
```
Performs bitwise OR on given vectors and returns result.
- **arr**: _list_ or _numpy.ndarray_
  List of features to perform the combination on.

**Returns:** a **_numpy.ndarray_** result of **_numpy.bitwise\_or(a,b)_** on given columns in **_arr_**.


___

### CategoricalClassification.generate\_correlated
```python
CategoricalClassification.generate_correlated(X, feature_indices, r=0.8)
```
Generates and adds new columns to given dataset **X**, correlated to the selected features, by a Pearson correlation coefficient of **r**. For vectors with mean 0, their correlation equals the cosine of their angle.  

- **X**: _list_ or _numpy.ndarray_:
  Dataset to perform the combinations on.
- **feature_indices**: _int_ or _list_ or _numpy.ndarray_:
  Index of feature (column) or list of feature (column) indices to generate correlated features to.
- **r**: _float_, default=0.8:
  Desired correlation coefficient.

**Returns:** a **_numpy.ndarray_** dataset X with added correlated features.

---

### CategoricalClassification.generate\_duplicates
```python
CategoricalClassification.generate_duplicates(X, feature_indices)
```

Duplicates selected feature (column) indices, and adds the duplicated columns to the given dataset **X**.

- **X**: _list_ or _numpy.ndarray_:
  Dataset to perform the combinations on.
- **feature_indices**: _int_ or _list_ or _numpy.ndarray_:
  Index of feature (column) or list of feature (column) indices to duplicate.

**Returns:** a **_numpy.ndarray_** dataset X with added duplicated features.

---
### CategoricalClassification.generate\_labels
```python
CategoricalClassification.generate_nonlinear_labels(X, n=2, p=0.5, k=2, decision_function=None, class_relation='linear', balance=False)
```

Generates a vector of labels. Labels are (currently) generated as either a linear, nonlinear, or custom defined function. It generates classes using a decision boundary generated by the linear, nonlinear, or custom defined function.

- **X**: _list_ or _numpy.ndarray_:
  Dataset to generate labels for.
- **n**: _int_, default=2:
  Number of classes.
- **p**: _float_ or _list_, default=0.5:
  Class distribution.
- **k**: _int_ or _float_, default=2:
  Constant to be used in the linear or nonlinear combination used to set class values.
- **decision_function**: _function_, default: None
  Custom defined function to use for setting class values. **Must** accept dataset X as input and return a _list_ or _numpy.ndarray_ decision boundary.
- **class_relation**: _str_, either _'linear'_, _'nonlinear'_, or _'cluster'_ default='linear':
  Sets relationship type between class label and sample, by calculating a decision boundary with linear or nonlinear combinations of features in X, or by clustering the samples in X.
- **balance**: _boolean_, default=False:
  Whether to naievly balance clusters generated by KMeans clustering.

 **Returns**: **_numpy.ndarray_** y of class labels.
 
---

### CategoricalClassification.\_cluster\_data
```python
CategoricalClassification._cluster_data(X, n, p=1.0, balance=False)
```
Clusters given data using KMeans clustering.

- **X**: _list_ or _numpy.ndarray_:
  Dataset to cluster.
- **n**: _int_:
  Number of clusters.
- **p**: _float_ or _list_ or _numpy.ndarray_:
  To be used when balance=True, sets class distribution - number of samples per cluster.
- **balance**: _boolean_, default=False:
  Whether to naievly balance clusters generated by KMeans clustering.

**Returns**: **_numpy.ndarray_** cluster_labels of clustering labels.
___

### CategoricalClassification.generate\_noise
```python
CategoricalClassification.generate_noise( X, y, p=0.2, type="categorical", missing_val=float('-inf'))
```

Generates categorical noise or simulates missing data on a given dataset. 

- **X**: _list_ or _numpy.ndarray_:
  Dataset to generate noise for.
- **y**: _list_ or _numpy.ndarray_:
  Labels of samples in dataset X. **Required** for generating categorical noise.
- **p**: _float_, p <=1.0, default=0.2:
  Amount of noise to generate.
- **type**: _str_, either _"categorical"_ or _"missing"_, default="categorical":
  Type of noise to generate.
- **missing_val**: default=float('-inf'):
  Value to simulate missing values with. Non-numerical values may cause issues with algorithms unequipped to handle them.

**Returns**: **_numpy.ndarray_** X with added noise.

---

### CategoricalClassification.downsample\_dataset
```python
CategoricalClassification.downsample_dataset(X, y, N=None, seed=42, reshuffle=False):
```

Downsamples given dataset according to N or the number of samples in minority class, resulting in a balanced dataset.

- **X**: _list_ or _numpy.ndarray_:
  Dataset to downsample.
- **y**: _list_ or _numpy.ndarray_:
  Labels corresponding to X.
- **N**: _int_, optional:
  Optional number of samples per class to downsample to.
- **seed**: _int_, default=42:
  Seed for random state of resample function.
- **reshuffle**: _boolean_, default=False:
  Reshuffle the dataset after downsampling.

**Returns:** Balanced, downsampled **_numpy.ndarray_** X and **_numpy.ndarray_** y.

---

### CategoricalClassification.print\_dataset
```python
CategoricalClassification.print_dataset(X, y)
```
Prints given dataset in a readable format.

- **X**: _list_ or _numpy.ndarray_:
  Dataset to print.
- **y**: _list_ or _numpy.ndarray_:
  Class labels corresponding to samples in given dataset.

---

### CategoricalClassification.summarize
```python
CategoricalClassification.summarize()
```
Prints stored dataset information dictionary in a digestible manner.
