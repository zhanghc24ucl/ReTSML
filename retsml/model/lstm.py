from torch import nn
from .base import ModelConfig


class MultiVariableLSTM(nn.Module):
    def __init__(self, config: ModelConfig, input_size, output_size):
        super(MultiVariableLSTM, self).__init__()
        self.dim = (input_size, output_size)

        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            batch_first=True  # Input Shape: (batch_size, seq_len, input_size)
        )
        self.fc = nn.Linear(config.hidden_size, output_size)  # 输出三个变量的预测值

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        # (batch_size, output_size)
        return out
