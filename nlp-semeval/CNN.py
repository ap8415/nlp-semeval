import torch.nn as nn
import torch.nn.functional as F

from PytorchModel import PytorchModel


class CNN(PytorchModel):
    """
    Convolutional neural network architecture used for the classification.
    """

    def __init__(self, vocab_size, embedding_dim, out_channels, window_size, output_dim, dropout=0.5):
        super(CNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv = nn.Conv2d(in_channels=1, out_channels=out_channels,
                              kernel_size=(window_size, embedding_dim))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fully_connected = nn.Linear(out_channels, output_dim)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        # print(x)
        x = self.embedding(x)
        x = x.unsqueeze(1)  # Add channel for convolution

        conv = self.conv(x)
        conv = self.relu(conv)
        conv = conv.squeeze(3)

        pooled = F.max_pool1d(conv, conv.shape[2])
        pooled = pooled.squeeze(2)
        dropped = self.dropout(pooled)

        outputs = self.fully_connected(dropped)
        predictions = self.logsoftmax(outputs)
        return predictions

    @staticmethod
    def default_instance():
        return CNN(1, 1, 1, 1, 1)

    def from_params(self, params):
        # out-channels is equal to embedding-dim for the CNN
        return CNN(params['vocab-size'], params['embedding-dim'], params['embedding-dim'], params['window-size'],
                   params['output-dim'])

    def get_param_grid(self):
        return {
            'window-size': [1, 2, 3, 4],  # up to 4-gram
            'embedding-dim': [30, 50, 75, 100, 150],  # note: out-channels is always equal to embedding-dim
        }
        # TODO: add extra layers maybe?

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.bias.data)
            nn.init.xavier_uniform(m.weight.data)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Embedding):
            nn.init.xavier_uniform(m.weight.data)


