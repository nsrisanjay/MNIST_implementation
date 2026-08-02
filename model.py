import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 1st convolution layer
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=0
        )

        self.reLu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(
            kernel_size=2,stride=2
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0
        )
        self.reLu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(
            stride=2,kernel_size=2
        )
        self.flatten = nn.Flatten(start_dim=1,end_dim=-1)
        self.linear1 = nn.Linear(in_features=1600,out_features=512)
        self.reLu3 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=512,out_features=128)
        self.reLu4 = nn.ReLU()
        self.linear3 = nn.Linear(in_features=128,out_features=10)

    def forward(self, data):
        data = self.conv1(data)
        data = self.reLu1(data)
        data = self.pool1(data)
        data = self.conv2(data)
        data = self.reLu2(data)
        data = self.pool2(data)
        data = self.flatten(data)
        data = self.linear1(data)
        data = self.reLu3(data)
        data = self.linear2(data)
        data = self.reLu4(data)
        dat = self.linear3(data)
        return data

