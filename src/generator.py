import numpy as np
from scipy.linalg import qr
from scipy.stats import norm

class CategoricalClassification:

    def __init__(self):
        self.dataset_info = ''

    def generate_data(self, n_features: int, n_samples: int, cardinality=5, structure=None, ensure_rep: bool = False, seed: int = 42) -> np.ndarray:

        """
        Generates dataset based on parameters
        :param n_features: number of generated features
        :param n_samples: number of generated samples
        :param cardinality: default cardinality of the dataset
        :param structure: structure of the dataset
        :param ensure_rep: flag, ensures all given values represented
        :param seed: sets seed of numpy random
        :return: X, 2D dataset
        """


        d = {'n_features': n_features, 'ix': (n_features - 1), 'n_samples': n_samples, 'cardinality': cardinality,
             'seed': seed}
        s = '''\
                Number of original features: {n_features}, at columns [0, ..., {ix}]\n\
                Number of samples: {n_samples}\n\
                Cardinality: {cardinality}\n\
                Random seed: {seed}\
            '''.format(**d)

        self.dataset_info = s

        np.random.seed(seed)
        X = np.empty([n_features, n_samples])

        if structure == None:

            for i in range(n_features):
                x = self._generate_feature(cardinality, n_samples, ensure_rep=ensure_rep)
                X[i] = x

        else:

            ix = 0
            for data in structure:

                if not isinstance(data[0], (list, np.ndarray)):
                    feature_ix = data[0]
                    feature_cardinality = data[1]
                    if ix < feature_ix:
                        for i in range(ix, feature_ix):
                            x = self._generate_feature(cardinality, n_samples, ensure_rep=ensure_rep)
                            X[ix] = x
                            ix += 1

                    if not isinstance(feature_cardinality, (list, np.ndarray)):
                        x = self._generate_feature(feature_cardinality, n_samples, ensure_rep=ensure_rep)
                    else:
                        value_domain = feature_cardinality[0]
                        value_frequencies = feature_cardinality[1]
                        x = self._generate_feature(value_domain, n_samples, ensure_rep=ensure_rep, p=value_frequencies)
                    X[ix] = x
                    ix += 1

                else:
                    feature_ixs = data[0]
                    feature_cardinality = data[1]
                    for feature_ix in feature_ixs:
                        if ix < feature_ix:
                            for i in range(ix, feature_ix):
                                x = self._generate_feature(cardinality, n_samples, ensure_rep=ensure_rep)
                                X[ix] = x
                                ix += 1

                        if not isinstance(feature_cardinality, (list, np.ndarray)):
                            x = self._generate_feature(feature_cardinality, n_samples, ensure_rep=ensure_rep)
                        else:
                            value_domain = feature_cardinality[0]
                            value_frequencies = feature_cardinality[1]
                            x = self._generate_feature(value_domain, n_samples, ensure_rep=ensure_rep, p=value_frequencies)
                        X[ix] = x
                        ix += 1

            if ix < n_features:
                for i in range(ix, n_features):
                    x = self._generate_feature(cardinality, n_samples, ensure_rep=ensure_rep)
                    X[i] = x

        return X.T

    def _generate_feature(self, v, size: int, ensure_rep: bool = False, p=None) -> np.ndarray:
        """
        Generates feature vector of length size. Default probability density distribution is approx. normal, centred around randomly picked value.
        :param v: either int for cardinality, or list of values
        :param size: length of feature vector
        :param ensure_rep: ensures all values are represented at least once in the feature vector
        :param p: list of probabilities of each value
        :return:
        """
        if not isinstance(v, (list, np.ndarray)):
            v = np.arange(0, v, 1)



        if p is None:
            v_shift = v - v[np.random.randint(len(v))]
            p = norm.pdf(v_shift, scale=3)
        
        p = p / p.sum()

        if ensure_rep and len(v) < size:
            sampled_values = np.random.choice(v, size=(size - len(v)), p=p)
            sampled_values = np.append(sampled_values, v)
        else:
            sampled_values = np.random.choice(v, size=size, p=p)

        np.random.shuffle(sampled_values)
        return sampled_values

    def generate_combinations(self, X, feature_indices, combination_function=None, combination_type='linear'):
        """
        Generates linear, nonlinear, or custom combinations within feature vectors in given dataset X
        :param X: dataset
        :param feature_indices: indexes of features to be in combination
        :param combination_function: optional custom function for combining feature vectors
        :param combination_type: string flag, either liner or nonlinear, defining combination type
        :return: X with added resultant feature
        """
        d = {'ixs': feature_indices, 'combination_ix': len(X[0])}

        selected_features = X[:, feature_indices]

        if combination_function is None:
            if combination_type == 'linear':
                combination_function = lambda x: np.sum(x, axis=1)
                d['func'] = 'linear'
            elif combination_type == 'nonlinear':
                combination_function = lambda x: np.sin(np.sum(x, axis=1))
                d['func'] = 'nonlinear'
        else:
            d['func'] = 'user defined'

        combination_result = combination_function(selected_features)

        s = '''
                Columns {ixs} are {func} combinations, result in column {combination_ix}\
            '''.format(**d)

        self.dataset_info += s

        return np.column_stack((X, combination_result))

    def generate_correlated(self, X, feature_indices, r=0.8):

        """
        Generates correlated features using the given feature indices. Correlation is based on cosine of angle between vectors with mean 0.
        :param X: dataset
        :param feature_indices: indices of features to generate correlated feature to
        :param r: (Pearson) correlation factor
        :return: X with generated correlated  features
        """

        d = {'ixs': feature_indices, 'corr': r, 'f0': len(X[0]), 'fn': (len(X[0]) + len(feature_indices) - 1)}

        selected_features = X[:, feature_indices]
        transposed = np.transpose(selected_features)
        correlated_features = []

        for t in transposed:
            theta = np.arccos(r)
            t_standard = (t - np.mean(t)) / np.std(t)

            rand = np.random.normal(0, 1, len(t_standard))
            rand = (rand - np.mean(rand)) / np.std(rand)

            M = np.column_stack((t_standard, rand))
            M_centred = (M - np.mean(M, axis=0))

            Id = np.eye(len(t))
            Q = qr(M_centred[:, [0]], mode='economic')[0]
            P = np.dot(Q, Q.T)
            orthogonal_projection = np.dot(Id - P, M_centred[:, 1])
            M_orthogonal = np.column_stack((M_centred[:, 0], orthogonal_projection))

            Y = np.dot(M_orthogonal, np.diag(1 / np.sqrt(np.sum(M_orthogonal ** 2, axis=0))))
            corr = Y[:, 1] + (1 / np.tan(theta)) * Y[:, 0]

            correlated_features.append(corr)

        correlated_features = np.transpose(correlated_features)

        s = '''
                Columns [{f0}, ..., {fn}] are correlated to {ixs} with a factor of {corr}\
            '''.format(**d)

        self.dataset_info += s

        return np.column_stack((X, correlated_features))

    def generate_duplicates(self, X, feature_indices):
        """
        Generates duplicate features
        :param X: dataset
        :param feature_indices: indices of features to duplicate
        :return: dataset with duplicated features
        """
        d = {'ixs': feature_indices, 'f0': len(X[0]), 'fn': (len(X[0]) + len(feature_indices) - 1)}

        selected_features = X[:, feature_indices]

        s = '''
                Columns [{f0}, ..., {fn}] are duplicates of {ixs}\
            '''.format(**d)


        self.dataset_info += s

        return np.column_stack((X, selected_features))

    def generate_labels(self, X, n=2, p=0.5, k=2, decision_function=None, class_relation='linear'):
        """
        Generates labels for dataset X
        :param X: dataset
        :param n: number of class labels
        :param p: class distribution
        :param k: constant
        :param decision_function: optional user-defined decision function
        :class_relation: string, either 'linear' or 'nonlinear'
        :return: array of labels, corresponding to dataset X
        """

        n_samples, n_features = X.shape
        d = {'classn': n, 'nfeatures': n_features}

        if decision_function is None:
            if class_relation == 'linear':
                decision_function = lambda x: np.sum(2 * x + 3, axis=1)
                d['type'] = 'linear, with constant ' + str(k)
            elif class_relation == 'nonlinear':
                decision_function = lambda x: np.sum(k * np.sin(x) + k * np.cos(x), axis=1)
                d['type'] = 'nonlinear, with constant ' + str(k)
        else:
            d['type'] = 'user defined'

        if n > 2:
            if type(p) != list:
                p = 1 / n
                percentiles = [p * 100]
                for i in range(1, n - 1):
                    percentiles.append(percentiles[i - 1] + (p * 100))

                decision_boundary = decision_function(X)
                p_points = np.percentile(decision_boundary, percentiles)

                y = np.zeros_like(decision_boundary, dtype=int)
                for p_point in p_points:
                    y += (decision_boundary > p_point)

        else:
            decision_boundary = decision_function(X)
            p_point = np.percentile(decision_boundary, p * 100)
            y = np.where(decision_boundary > p_point, 1, 0)

        s = '''
                Sample-label relationship is {type}, with {classn} target labels.\n\
                Total number of features generated: {nfeatures}\
            '''.format(**d)

        self.dataset_info += s

        return y

    def generate_noise(self, X, y, p=0.2, type="categorical", missing_val=float('-inf')):

        """
        Simulates noise on given dataset X
        :param X: dataset to apply noise to
        :param y: required target labels for categorical noise generation
        :param p: amount of noise to apply. Defaults to 0.2
        :param type: type of noise to apply, either categorical or missing
        :param missing_val: value to simulate missing values. Defaults to float('-inf')
        :return: X with noise applied
        """


        if type == "categorical":
            label_values, label_count = np.unique(y, return_counts=True)
            n_labels = len(label_values)

            inds = y.argsort()
            y_sort = y[inds]
            X_sort = X[inds]

            Xs_T = X_sort.T
            n = Xs_T.shape[1]
            n_flip = int(n * p)

            for feature in Xs_T:
                unique_per_label = {}

                for i in range(n_labels):
                    if i == 0:
                        unique = np.unique(feature[:label_count[i]])
                        unique_per_label[label_values[i]] = set(unique)
                    else:
                        unique = np.unique(feature[label_count[i - 1]:label_count[i - 1] + label_count[i] - 1])
                        unique_per_label[label_values[i]] = set(unique)

                ixs = np.random.choice(n, n_flip, replace=False)

                for ix in ixs:
                    current_label = y_sort[ix]
                    possible_labels = np.where(label_values != current_label)[0]

                    # find all unique values from labels != current label
                    values = set()
                    for key in possible_labels:
                        values = values.union(unique_per_label[key])

                    # remove any overlapping values, ensuring replacement values are unique & from a target label !=
                    # current label
                    for val in unique_per_label[current_label] & values:
                        values.remove(val)

                    if len(values) > 0:
                        val = np.random.choice(list(values))

                    else:
                        key = possible_labels[np.random.randint(len(possible_labels))]
                        values = unique_per_label[key]
                        val = np.random.choice(list(values))

                    feature[ix] = val

            rev_ind = inds.argsort()
            X_noise = Xs_T.T
            X_noise = X_noise[rev_ind]

            return X_noise

        elif type == "missing":
            X_noise = np.copy(X)
            Xn_T = X_noise.T
            n = Xn_T.shape[1]
            n_missing = int(n * p)
            print("n to delete:", n_missing)

            for feature in Xn_T:
                ixs = np.random.choice(n, n_missing, replace=False)

                for ix in ixs:
                    feature[ix] = missing_val

            return Xn_T.T

    def print_dataset(self, X, y):
        """
        Prints given dataset
        :param X: dataset
        :param y: labels
        :return:
        """

        n_samples, n_features = X.shape
        n = 0
        for arr in X:
            print('[', end='')
            for i in range(n_features):
                if i == n_features - 1:
                    print(arr[i], end='')
                else:
                    print(arr[i], end=', ')
            print("], Label: {}".format(y[n]))
            n += 1
