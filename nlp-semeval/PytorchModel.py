from abc import abstractmethod

import torch

import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from Model import Model
from torch.autograd import Variable

from ParamGrid import ParamGrid


class PytorchModel(nn.Module, Model):

    def __init__(self):
        super(PytorchModel, self).__init__()
        self.vocabulary = []
        self.w2i = []
        self.max_len = -1

    @staticmethod
    @abstractmethod
    def weights_init(m):
        pass

    def train_model(self, corpus, labels):
        f1_scores = []
        cross_validation_folds = PytorchModel.cross_validation(corpus, labels)
        for training_corpus, training_labels, validation_corpus, validation_labels in cross_validation_folds:
            self.train_fold(training_corpus, training_labels, 20)
            f1, _, _, _, confusion = self.evaluate(validation_corpus, validation_labels)
            f1_scores.append(f1)
            print(confusion)
            self.apply(self.weights_init)  # Restore the model to default

        total_f1 = sum(f1_scores) / len(f1_scores)
        print(total_f1)

        self.train_fold(corpus, labels, 50)
        return self, total_f1

    def train_fold(self, corpus, labels, total_epochs):
        optimizer = optim.SGD(self.parameters(), lr=0.05)
        loss_fn = nn.NLLLoss()

        sentence_tensor, label_tensor = PytorchModel.get_model_inputs(corpus, labels, self.w2i)

        tensor_loader = DataLoader(TensorDataset(sentence_tensor), batch_size=32)
        label_loader = DataLoader(TensorDataset(label_tensor), batch_size=32)

        epochs = 0
        while epochs < total_epochs:
            self.train()

            tensor_iter = iter(tensor_loader)
            label_iter = iter(label_loader)

            i = 0
            curr_loss = 0.0
            while True:
                # Simulate end-of-data with try-catch block on iterators
                try:
                    inputs = next(tensor_iter)[0]
                    labels = next(label_iter)[0]

                    optimizer.zero_grad()

                    predictions = self(inputs).squeeze(1)

                    loss = loss_fn(predictions, labels)
                    loss.backward()
                    optimizer.step()

                    self.eval()
                    curr_loss += loss.item()
                    i += 1
                except StopIteration:
                    break
            epochs += 1
            curr_loss = curr_loss / i  # Average over all batches
            print('Average loss at epoch %d: %.4f' % (epochs, curr_loss))

        self.eval()

    def predict_on_corpus(self, test_corpus):
        # If no vocabulary is present, model hasn't been trained. Abort.
        assert len(self.vocabulary) > 0
        test_tensor = self.get_test_inputs(test_corpus, self.w2i)
        predictions = torch.argmax(self(test_tensor), dim=1)
        return predictions.tolist()

    def optimize(self, corpus, labels):
        """
        Performs grid search within the model's possible parameters, given by the get_param_grid() function,
        and returns the best-performing model.
        """
        vocab, w2i = PytorchModel.build_vocabulary_and_w2i(corpus)

        # Build parameter grid
        default_params = {
            'vocab-size': len(vocab),
            'output-dim': len(set(labels)),
            'max-len': max([len(sentence.split()) for sentence in corpus])
        }
        optimizable_params = self.get_param_grid()
        param_grid = ParamGrid(optimizable_params, default_params)

        best_model = None
        best_f1_score = 0.0
        for params in param_grid:
            model = self.from_params(params)
            model, f1_score = model.train_model(corpus, labels)
            if f1_score > best_f1_score:
                best_model = model

        return best_model

    @abstractmethod
    def from_params(self, params):
        """
        Builds a new instance of this model from the given parameters.
        :param params:
        :return:
        """
        pass

    @staticmethod
    def tokenize(corpus):
        return [x.split() for x in corpus]

    @staticmethod
    def build_vocabulary_and_w2i(corpus):
        vocabulary = []
        for sentence in PytorchModel.tokenize(corpus):
            for token in sentence:
                if token not in vocabulary:
                    vocabulary.append(token)

        w2i = {w: idx + 1 for (idx, w) in enumerate(vocabulary)}
        w2i['<pad>'] = 0

        return vocabulary, w2i

    @staticmethod
    def get_model_inputs(corpus, labels, w2i):
        vectorized_sentences = [[w2i[tok] for tok in sentence if tok in w2i]
                                for sentence in PytorchModel.tokenize(corpus)]
        max_len = max([len(sentence) for sentence in vectorized_sentences])
        sentence_tensor = Variable(torch.zeros((len(vectorized_sentences), max_len))).long()
        sentence_lengths = [len(sent) for sent in vectorized_sentences]

        for idx, (sent, sent_length) in enumerate(zip(vectorized_sentences, sentence_lengths)):
            sentence_tensor[idx, : sent_length] = torch.LongTensor(sent)
        label_tensor = torch.LongTensor(labels)
        return sentence_tensor, label_tensor

    def get_test_inputs(self, test_corpus, w2i):
        vectorized_sentences = [[w2i[tok] for tok in sentence if tok in w2i]
                                for sentence in PytorchModel.tokenize(test_corpus)]
        vectorized_sentences = [sentence[:self.max_len] for sentence in vectorized_sentences]
        sentence_tensor = Variable(torch.zeros((len(vectorized_sentences), self.max_len))).long()
        sentence_lengths = [len(sent) for sent in vectorized_sentences]

        for idx, (sent, sent_length) in enumerate(zip(vectorized_sentences, sentence_lengths)):
            sentence_tensor[idx, : sent_length] = torch.LongTensor(sent)
        return sentence_tensor
