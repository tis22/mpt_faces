from torch import nn
import torch.nn.functional as F

# NOTE: This will be the network architecture.


class Net(nn.Module):
    def __init__(self, nClasses):
        super().__init__()

        # TODO: Implement module constructor.
        # Define network architecture as needed
        # Input imags will be 3 channels 256x256 pixels.
        # Output must be a nClasses Tensor.

        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)

        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, nClasses)

    def forward(self, x):
        # TODO:
        # Implement forward pass
        #  x is a BATCH_SIZEx3x256x256 Tensor
        #  return value must be a BATCH_SIZExN_CLASSES Tensor

        # conv1 + ReLU + MaxPooling
        x = self.pool(F.relu(self.conv1(x)))

        # conv2 + ReLU + MaxPooling
        x = self.pool(F.relu(self.conv2(x)))

        # conv3 + ReLU + MaxPooling
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten output for fully connected layer
        x = x.view(-1, 128 * 32 * 32)

        # Fully connected layer, ReLU activated
        x = F.relu(self.fc1(x))

        # Output layer
        x = self.fc2(x)

        return x
