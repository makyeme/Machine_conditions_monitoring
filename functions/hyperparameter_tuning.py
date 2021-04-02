import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from functions.functions import read_data

SR = 16000


class ParamaterTuning:
    """
    Class that will tune the parameters
    """

    def __init__(self):
        pass

    def read_data(self):
        self.x_train, self.x_val, self.y_train, self.y_val = read_data(cv=True)

    def tune_params(self, pickled=True):
        print('Started tuning functions')
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=50, stop=250, num=10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(50, 100, num=6)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [3, 4, 6, 8, 10, 12, 14]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4, 6, 8]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]# Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        rf = RandomForestClassifier()
        # Random search of parameters, using 3 fold cross validation,
        # search across 100 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=30, cv=3, verbose=2,
                                       random_state=42, n_jobs=6)  # Fit the random search model
        print('Start tuning rfc')
        rf_random.fit(self.x_train[::2], self.y_train[::2])
        self.best_params_ = rf_random.best_params_
        print('Done tuning, saving...')
        if pickled:
            rfc = RandomForestClassifier(**self.best_params_)
            rfc.fit(self.x_train, self.y_train)
            pickle.dump(rfc, open('models/rfc_tuned', 'wb'))
            print('Done saving')
            return rfc
