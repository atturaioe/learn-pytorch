import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv1_bn = nn.BatchNorm2d(6)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.conv2_bn = nn.BatchNorm2d(12)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc1_bn = nn.BatchNorm1d(120)

        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.fc2_bn = nn.BatchNorm1d(60)

        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # 1st layer
        t = self.conv1_bn(self.conv1(t))
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # 2nd layer
        t = self.conv2_bn(self.conv2(t))
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # 3rd layer
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1_bn(self.fc1(t))
        t = F.relu(t)

        # 4th layer
        t = self.fc2_bn(self.fc2(t))
        t = F.relu(t)

        # 5th layer
        t = self.out(t)

        # torch.nn.functional.cross_entropy applies softmax function
        return t
