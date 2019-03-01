import csv

import torch
import torch.nn as nn
from torch.autograd import Variable

from data_processing import *
from feature_extraction import FeatureExtractor

from CNN import CNN
from GRU import GRU
from LSTM import LSTM
from LinearSVModel import LinearSVModel
from RandomForestModel import RandomForestModel
from LogisticRegressionModel import LogisticRegressionModel

"""
Main file of the OffensEval classifier.
For each subtask, and each type of implemented model, it:
 - searches for the best-performing parameters by using a fourth of the training set as holdout.
 - it then runs each of the obtained models to obtain predictions on the OffensEval test set.
 - finally, it takes the best performing 3 models (on the holdout) and makes another prediction based on voting from the
 three models.
"""

subtasks = ['subtask_a', 'subtask_b', 'subtask_c']

test_sets = {'subtask_a': "./data/Test A Release/testset-taska.tsv",
             'subtask_b': "./data/Test B Release/testset-taskb.tsv",
             'subtask_c': "./data/Test C Release/testset-taskc.tsv"}

val_to_pred_map = {
    'subtask_a': {0: 'NOT', 1: 'OFF'},
    'subtask_b': {0: 'UNT', 1: 'TIN'},
    'subtask_c': {0: 'OTH', 1: 'GRP', 2: 'IND'}
}

training_data_file_path = "./data/start-kit/training-v1/offenseval-training-v1.tsv"

for subtask in subtasks:
    training_data = preprocess(training_data_file_path, subtask=subtask)
    feature_extractor = FeatureExtractor()
    corpus, labels = parse(training_data, subtask=subtask)
    feature_extractor.tfidf_fit_transform(corpus)

    models = [CNN.default_instance(), LSTM.default_instance(), GRU.default_instance(),
              LinearSVModel(feature_extractor), RandomForestModel(feature_extractor),
              LogisticRegressionModel(feature_extractor)]

    optimized_models = []

    for model in models:
        optimized_models.append(model.optimize(corpus, labels))

    # Get predictions on test sets
    test_data = preprocess_test_set(test_sets[subtask])
    test_corpus = get_corpus(test_data)

    for idx, model in enumerate(optimized_models):
        predictions = model.predict_on_corpus(test_corpus)

        with open('%s-prediction-model%d.csv' % (subtask, idx), 'w') as predictions_file:
            preds = []

            for i in range(0, len(test_data)):
                id = test_data[i]['id']
                pred = val_to_pred_map[subtask][predictions[i]]
                preds.append((id, pred))

            writer = csv.writer(predictions_file)
            writer.writerows(preds)
