import torch.nn as nn

class ViolenceRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, rnn_type='LSTM'):
        super().__init__()
        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        _, (hn, _) = self.rnn(x)
        return self.fc(hn[-1])
