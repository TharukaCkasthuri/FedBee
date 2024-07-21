"""
Copyright (C) [2023] [Tharuka Kasthuriarachchige]

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Paper: [FeDABoost: AdaBoost Enhanced Federated Learning]
Published in: 
"""

import torch
from torch import nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self, num_classes):
        super(SimpleNet, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(14*14*8, self.num_classes)
        self.bn1 = nn.BatchNorm2d(8)
        self.dropout = nn.Dropout(p=0.5)
        self.track_layers = {"conv1": self.conv1, 
                             "fc1": self.fc1}


    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)  # Max pooling over a (2, 2) window
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(x)  
        x = self.fc1(x)
        return x

    def process_x(self, raw_x_batch):
        return torch.tensor(raw_x_batch, dtype=torch.float32)

    def process_y(self, raw_y_batch):
        return torch.tensor(raw_y_batch, dtype=torch.long)
