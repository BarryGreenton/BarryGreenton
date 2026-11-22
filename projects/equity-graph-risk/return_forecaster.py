import math
import torch
from torch import nn

class MultiVariateLSTM_Transformer(nn.Module):

    def __init__(self, input_size=23, lstm_hidden_size=64, lstm_layers=2, trans_layers=2, trans_heads=4, trans_ffn_hidden=128, dropout=0.1, output_size=1, output_window=10):
        super(MultiVariateLSTM_Transformer, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.output_window = output_window
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, lstm_layers, batch_first=True, dropout=dropout)
        self.pos_encoder = PositionalEncoding(d_model=lstm_hidden_size, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=lstm_hidden_size, nhead=trans_heads, dim_feedforward=trans_ffn_hidden, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=trans_layers)
        self.fc = nn.Linear(lstm_hidden_size, output_size * output_window)

    def forward(self, x):
        batch_size = x.size(0)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.transpose(0, 1)
        lstm_out = self.pos_encoder(lstm_out)
        transformer_out = self.transformer_encoder(lstm_out)
        last_output = transformer_out[-1]
        output = self.fc(last_output)
        output = output.view(batch_size, self.output_window, -1)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
