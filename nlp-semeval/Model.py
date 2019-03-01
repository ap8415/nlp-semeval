from sklearn.metrics import *
from abc import ABC, abstractmethod

no_of_outputs = {'subtask_a': 2, 'subtask_b': 2, 'subtask_c': 3}

class Model(ABC):

    @abstractmethod
    def predict_on_corpus(self, test_corpus):
        """
        :rtype: List of labels
        """
        pass

    @abstractmethod
    def train_model(self, training_corpus, training_labels):
        """
        Performs training with 5-fold cross validation on the training set to determine the power of the model,
        then train on the whole training set.
        :return: A tuple, consisting of the model itself, and the averaged macro f1-score for the cross-validation
        folds.
        """
        pass

    def evaluate(self, validation_corpus, validation_labels):
        predicted_labels = self.predict_on_corpus(validation_corpus)

        acc = accuracy_score(validation_labels, predicted_labels)
        prec = precision_score(validation_labels, predicted_labels, average='macro')
        recall = recall_score(validation_labels, predicted_labels,  average='macro')
        macro_f1 = f1_score(validation_labels, predicted_labels,  average='macro')
        confusion = confusion_matrix(validation_labels, predicted_labels)
        # TODO: Compute loss

        return macro_f1, acc, prec, recall, confusion

    @abstractmethod
    def optimize(self, training_corpus, training_labels):
        """
        Performs grid search on the given parameters, and returns the best-performing model.
        :param training_corpus:
        :param training_labels:
        :param params:
        :return:
        """
        pass

    @abstractmethod
    def get_param_grid(self):
        pass

    @staticmethod
    @abstractmethod
    def default_instance():
        pass

    @staticmethod
    def cross_validation(corpus, labels):
        return [(Model.n_fold_cross_validation(5, i+1, corpus, labels)) for i in range(0, 5)]

    @staticmethod
    def n_fold_cross_validation(n, k, corpus, labels):
        """
        Splits the corpus into n folds, and returns the k'th fold as the validation set, and the rest as the
        training data.
        """
        lower_bound = int((k-1) * len(corpus) / n)
        upper_bound = int(k * len(corpus) / n)
        training_corpus = corpus[:lower_bound] + corpus[upper_bound:]
        validation_corpus = corpus[lower_bound:upper_bound]
        training_labels = labels[:lower_bound] + labels[upper_bound:]
        validation_labels = labels[lower_bound:upper_bound]
        return training_corpus, training_labels, validation_corpus, validation_labels
