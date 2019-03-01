from sklearn.svm import LinearSVC

from ScikitModel import ScikitModel


class LinearSVModel(ScikitModel):

    def __init__(self, feature_extractor, params={}):
        params = {**params, 'verbose': 10}
        model = LinearSVC(**params)
        super(LinearSVModel, self).__init__(model, feature_extractor)

    def get_param_grid(self):
        return {'penalty': ['l1', 'l2']}

    @staticmethod
    def default_instance():
        return None
