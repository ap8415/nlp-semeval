from sklearn.ensemble import RandomForestClassifier

from ScikitModel import ScikitModel


class RandomForestModel(ScikitModel):

    def __init__(self, feature_extractor, params={}):
        params = {**params, 'verbose': 10}
        model = RandomForestClassifier(**params)
        super(RandomForestModel, self).__init__(model, feature_extractor)

    def get_param_grid(self):
        return {'n_estimators': [n for n in range(16, 160, 16)]}
