import torch

class CNN_NET(torch.nn.Module):
    def __init__(self):
        super(CNN_NET, self).__init__()
        self.cnn_layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, stride=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, padding=1, stride=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.ReLU(),
        )
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(7*7*32, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 10),
            torch.nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        out = self.cnn_layers(x)
        out = out.view(-1, 7*7*32)
        out = self.fc_layers(out)
        return out