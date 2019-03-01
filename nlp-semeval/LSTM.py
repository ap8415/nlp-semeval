import torch.nn as nn

from PytorchModel import PytorchModel


class LSTM(PytorchModel):
    """
    Deep network with a LSTM layer.
    """

    def __init__(self, vocab_size, max_len, hidden_dim, embedding_dim, output_dim):
        super(LSTM, self).__init__()

        self.max_len = max_len
        self.hd = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(self.max_len * self.hd, output_dim)
        self.activation = nn.LogSoftmax()

    def forward(self, x):
        x = self.embedding(x)
        x, (_, _) = self.lstm(x)
        x = x.reshape(-1, self.max_len * self.hd)

        x = self.fc1(x)
        predictions = self.activation(x)

        return predictions

    @staticmethod
    def default_instance():
        return LSTM(1, 1, 1, 1, 1)

    def from_params(self, params):
        # out-channels is equal to embedding-dim for the CNN
        return LSTM(params['vocab-size'], params['max-len'], params['hidden-dim'], params['embedding-dim'],
                    params['output-dim'])

    def get_param_grid(self):
        return {
            'hidden-dim': [8, 16, 32, 64],  # tries various LSTM output sizes
            'embedding-dim': [30, 50, 75, 100, 150],
        }
        # TODO: add extra layers maybe?

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.bias.data)
            nn.init.xavier_uniform(m.weight.data)
        elif isinstance(m, nn.LSTM) or isinstance(m, nn.Embedding):
            nn.init.xavier_uniform(m.weight.data)

