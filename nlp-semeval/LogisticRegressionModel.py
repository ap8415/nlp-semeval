from sklearn.linear_model import LogisticRegression

from ScikitModel import ScikitModel


class LogisticRegressionModel(ScikitModel):

    def __init__(self, feature_extractor, params={}):
        params = {**params, 'verbose': 10}
        model = LogisticRegression(**params)
        super(LogisticRegressionModel, self).__init__(model, feature_extractor)

    def get_param_grid(self):
        return {
            'C': [(x * x) / 100 for x in range(5, 21)],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag']
        }
