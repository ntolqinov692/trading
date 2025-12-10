import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out,_ = self.lstm(x)
        return self.fc(out[:, -1])
