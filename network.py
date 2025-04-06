import torch
import torch.nn as nn

class ClassificadorRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, tipo='LSTM'):
        super().__init__()
        if tipo == 'RNN':
            self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        elif tipo == 'GRU':
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])  # usa o último timestep
