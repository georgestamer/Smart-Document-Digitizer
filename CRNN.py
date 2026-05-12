import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.rnn = nn.LSTM(
            input_size=128 * 8,
            hidden_size=256,
            num_layers=2,
            bidirectional=True
        )

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)  # (B, C, H, W)

        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)  # (B, W, C, H)
        x = x.reshape(b, w, c * h)

        x, _ = self.rnn(x)
        x = self.fc(x)

        return x