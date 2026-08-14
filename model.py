# import torch.nn as nn

# class CNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 1st convolution layer
#         self.conv1 = nn.Conv2d(
#             in_channels=1,
#             out_channels=32,
#             kernel_size=3,
#             stride=1,
#             padding=0
#         )

#         self.reLu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(
#             kernel_size=2,stride=2
#         )
#         self.conv2 = nn.Conv2d(
#             in_channels=32,
#             out_channels=64,
#             kernel_size=3,
#             stride=1,
#             padding=0
#         )
#         self.reLu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(
#             stride=2,kernel_size=2
#         )
#         self.flatten = nn.Flatten(start_dim=1,end_dim=-1)
#         self.linear1 = nn.Linear(in_features=1600,out_features=512)
#         self.reLu3 = nn.ReLU()
#         self.linear2 = nn.Linear(in_features=512,out_features=128)
#         self.reLu4 = nn.ReLU()
#         self.linear3 = nn.Linear(in_features=128,out_features=10)

#     def forward(self, data):
#         data = self.conv1(data)
#         data = self.reLu1(data)
#         data = self.pool1(data)
#         data = self.conv2(data)
#         data = self.reLu2(data)
#         data = self.pool2(data)
#         data = self.flatten(data)
#         data = self.linear1(data)
#         data = self.reLu3(data)
#         data = self.linear2(data)
#         data = self.reLu4(data)
#         data = self.linear3(data)
#         return data

import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Input: 1 × 32 × 32

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=2,
            stride=1,
            padding=1,
            padding_mode='replicate'
        )
        self.relu1 = nn.ReLU()
        # 64 × 33 × 33

        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=2,
            stride=1,
            padding=1,
            padding_mode='replicate'
        )
        self.relu2 = nn.ReLU()
        # 128 × 34 × 34

        self.maxpool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        # 128 × 17 × 17

        self.conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=192,
            kernel_size=2,
            stride=1,
            padding=1,
            padding_mode='replicate'
        )
        self.relu3 = nn.ReLU()
        # 192 × 18 × 18

        self.conv4 = nn.Conv2d(
            in_channels=192,
            out_channels=256,
            kernel_size=2,
            stride=1,
            padding=1,
            padding_mode='replicate'
        )
        self.relu4 = nn.ReLU()
        # 256 × 19 × 19

        self.maxpool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        # 256 × 9 × 9

        self.flatten1 = nn.Flatten()

        # 256 × 9 × 9 = 20736
        self.linear1 = nn.Linear(
            in_features=256 * 9 * 9,
            out_features=8192
        )
        self.relu5 = nn.ReLU()

        self.linear2 = nn.Linear(
            in_features=8192,
            out_features=2048
        )
        self.relu6 = nn.ReLU()

        self.linear3 = nn.Linear(
            in_features=2048,
            out_features=512
        )
        self.relu7 = nn.ReLU()

        self.linear4 = nn.Linear(
            in_features=512,
            out_features=10
        )

    def forward(self, data):

        data = self.conv1(data)
        data = self.relu1(data)

        data = self.conv2(data)
        data = self.relu2(data)

        data = self.maxpool1(data)

        data = self.conv3(data)
        data = self.relu3(data)

        data = self.conv4(data)
        data = self.relu4(data)

        data = self.maxpool2(data)

        data = self.flatten1(data)

        data = self.linear1(data)
        data = self.relu5(data)

        data = self.linear2(data)
        data = self.relu6(data)

        data = self.linear3(data)
        data = self.relu7(data)

        data = self.linear4(data)

        return data