from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def mnistDataset(save_dir = 'MNIST/', num_workers = 1, pin_memory = True, batch_size = 64, shuffle = False):

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    data_train = datasets.MNIST(root=save_dir, transform=transform, train=True, download=True)
    data_val = datasets.MNIST(root=save_dir, transform=transform, train=False, download=True)

    data_loader_train = DataLoader(dataset=data_train, num_workers=num_workers, pin_memory=pin_memory, batch_size=batch_size, shuffle=shuffle)
    data_loader_val = DataLoader(dataset=data_val, num_workers=num_workers, pin_memory=pin_memory, batch_size=batch_size, shuffle=shuffle)

    return data_loader_train, data_loader_val

if __name__ == '__main__':
    train_data = datasets.MNIST("MNIST/", train=True, transform=None,target_transform=None,download=True)
    test_data = datasets.MNIST("MNIST/", train=False, transform=None,target_transform=None,download=True)

    print('Number of samples in train_data is: ',len(train_data))
    print('Number of samples in test_data is: ',len(test_data))

    from matplotlib import pyplot as plt
    import numpy as np
    x = train_data.data[0]
    plt.imshow(x)
    plt.waitforbuttonpress()