from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from Model import Model
from data_processing import *

models = {
    'RandomForest': RandomForestClassifier,
    'LogisticRegression': LogisticRegression,
    # 'MLPClassifier': MLPClassifier,
    'LinearSVC': LinearSVC
}

model_param_grid = {
    'RandomForest': {
        'n_estimators': [n for n in range(16, 160, 16)],
    },
    'LogisticRegression': {
        'C': [(x * x) / 100 for x in range(5, 21)],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag']
    },
    'LinearSVC': {
        'penalty': ['l1', 'l2']
    }
}


class ScikitModel(Model):
    """
    Class that wraps around all the models we use from scikit-learn.
    TODO: say it implements the interface of all our models
    """
    def __init__(self, model, feature_extractor):
        self.model = model
        self.feature_extractor = feature_extractor

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def train_model(self, training_corpus, training_labels):
        self.fit(get_feature_vectors(training_corpus, self.feature_extractor), training_labels)
        # we don't use the F1 score for this kind of model as the fit for sklearn models does cross-validation already
        return self, 100.00

    def predict(self, X):
        return self.model.predict(X)

    def predict_on_corpus(self, test_corpus):
        return self.predict(get_feature_vectors(test_corpus, self.feature_extractor))

    def optimize(self, corpus, labels):
        param_grid = self.get_param_grid()
        grid_search = GridSearchCV(self.model, param_grid, n_jobs=4, verbose=1, scoring='f1_macro')
        grid_search.fit(get_feature_vectors(corpus, self.feature_extractor), labels)
        self.model = grid_search.best_estimator_
        return self

    @staticmethod
    def default_instance():
        return None # Not implemented for scikit


