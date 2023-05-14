import torch
import torch.nn as nn


class CustomMLPNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(in_features = 784, out_features = 128, bias = True)
        self.relu_1 = nn.ReLU(inplace = True)
        self.fc_2 = nn.Linear(in_features = 128, out_features = 128, bias = True)
        self.relu_2 = nn.ReLU(inplace = True)
        self.fc_3 = nn.Linear(in_features = 128, out_features = 10, bias = True)

    def forward(self, x):

        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.relu_1(x)
        x = self.fc_2(x)
        x = self.relu_2(x)
        output = self.fc_3(x)

        return output


class CustomCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, stride = 1, padding = 0)
        self.relu_1 = nn.ReLU(inplace = True)
        self.conv_2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 0)
        self.relu_2 = nn.ReLU(inplace = True)
        self.flatten = nn.Flatten()
        self.fc_3 = nn.Linear(in_features = 18432, out_features = 10, bias = True)

    def forward(self, x):

        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.flatten(x)

        output = self.fc_3(x)

        return output

if __name__ == '__main__':
    from torchvision import datasets
    mnist_dataset = datasets.MNIST('MNIST/', train=False, download= True)

    model = CustomCNN()

    input = mnist_dataset.data[0]
    input = torch.unsqueeze(input, 0)
    input = torch.unsqueeze(input, 0) / 255.0

    print(f'Input shape: {input.shape}')
    output = model(input)
    print(torch.sum(output[0]))
    print(f'Output shape: {output.shape}')

