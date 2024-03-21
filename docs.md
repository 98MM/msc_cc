# Documentation
---

### CategoricalClassification.dataset_info
```python
print(CategoricalClassification.dataset_info)
```
Stores a formatted string of operations made. Function _CategoricalClassification.generate\_data_ resets its contents. Each subsequent function call adds information to it.

---

### CategoricalClassification.generate_data
```python
CategoricalClassification.generate_data(n_features, n_samples, cardinality=5, ensure_rep=False, seed=42)
```
Generates dataset of shape **_(n_samples, n_features)_**, based on given parameters.

- **n\_features:** _int_
  The number of features in a generated dataset.
- **n\_samples:** _int_
  The number of samples in a generated dataset.
- **cardinality:** _int, list, numpy.ndarray_, default=5.
  Sets the cardinality and shape of a generated dataset.
  -  If **_int_**, the entire dataset has a cardinality of given int, with roughly normal density distribution of values, with a randomly selected value as a peak. The feature values will be integers, in range **\[0, cardinality\]**.
  -  **_list_** or **_numpy.ndarray_** offers more control and follows the format **\[_int_, _tuple_, _tuple_, ...\]**, where:
    - **_int_** represents the default cardinality of a generated dataset
    - **_tuple_** can either be:
      - **(_int_ or _list_, _int_)**: the first element represents the index or list of indexes of features. The second element their cardinality. Generated features will have a roughly normal density distribution of values, with a randomly selected value as a peak. The feature values will be integers, in range \[0, second element of tuple\].
      - **(_int_ or _list_, _list_)**: the first element represents the index or list of indexes of features. The second element offers two options:
        - **_list_**:  a list of values to be used in the feature or features,
        - **\[_list_, _list_\]**: where the first _list_ element represents a set of values the feature or features posses, the second the frequencies or probabilities of individual features.
- **ensure_rep:** _bool_, default=False:
  Control flag. If **_True_**, all possible values **will** appear in the feature.
- **seed**: _int_, default=42.
  Controls **_numpy.random.seed_**               

**Returns**: a **_numpy.ndarray_** dataset with **n\_features** features and **n\_samples** samples.

---

### CategoricalClassification.\_generate\_feature
```python
CategoricalClassification._generate_feature(v, size, ensure_rep=False, p=None)
```
Generates feature array of length **_size_**. Called by _CategoricalClassification.generate\_data_, by utilizing _numpy.random.choice_. If no probabilites array is given, the value density of the generated feature array will be roughly normal, with a randomly chosen peak. The peak will be chosen from the value array.

- **v**: _int_ or _list_:
  Range of values in feature array. If _int_ the function generates a value array in range _\[0, v\]_.
- **size**: _int_
  Length of generated feature array.
- **ensure_rep**: _bool_, default=False
  Control flag. If **_True_**, all possible values **will** appear in the feature array.
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
_Duplicate functions, will be reworked and combined into one._
```python
CategoricalClassification.generate_nonlinear_labels(X, n=2, p=0.5, k=2, decision_function=None, class_relation='linear')
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
- **class_relation**: _str_, either 'linear' or 'nonlinear', default='linear':
  Sets relationship type between class label and sample, by calculating a decision boundary with linear or nonlinear combinations of features in X.

 **Returns**: **_numpy.ndarray_** y of class labels.
 
---

### CategoricalCLassification.print\_dataset
```python
CategoricalClassification.print_dataset(X, y)
```
Prints given dataset in a readable format.

- **X**: _list_ or _numpy.ndarray_:
  Dataset to print.
- **y**: _list_ or _numpy.ndarray_:
  Class labels corresponding to samples in given dataset.
