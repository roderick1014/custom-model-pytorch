import torch
import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# This function checks if the directory specified by the input direction exists and if not, it creates it.
def mkdir(direction):
    if not os.path.exists(direction):
        os.makedirs(direction)

# This function returns the current date in the format '_YYYYMMDD'.
def getCurrentDate():
    cur_time = time.localtime()
    return f'_{cur_time.tm_year}{cur_time.tm_mon}{cur_time.tm_mday}'

# This function returns the current time in the format '_YYYYMMDD_HHMMSS'.
def getCurrentTime():
    cur_time = time.localtime()
    return f'_{cur_time.tm_year}{cur_time.tm_mon:02d}{cur_time.tm_mday:02d}_{cur_time.tm_hour:02d}{cur_time.tm_min:02d}{cur_time.tm_sec:02d}'

# This function displays num samples from the input data.
def displaySample(data, num):
    images, labels = iter(data).next()

    for idx in range(num):
        plt.imshow(images[idx - 1][0], cmap='gray')
        plt.axis('off')
        singleDividingLine(f' Label: {labels[idx - 1]}')
        plt.show()

# This function initializes the training process by displaying the configuration and checking if GPU is available.
def initMessege(args):
    messegeDividingLine('Initializing...')
    displayConfig(args)
    checkGPU(args.DEVICE)

# This function displays the configuration parameters.
def displayConfig(args):
    # vars() function returns all attributes in the class in a dict().
    print(f'Configuration: {vars(args)}')

# This function prints a dividing line along with the input messege.
def singleDividingLine(messege):
    print('=' * 40)
    print(messege)

# This function prints a dividing line above and below the input messege.
def messegeDividingLine(messege):
    print('=' * 40)
    print(messege)
    print('=' * 40)

# This function checks if the GPU is available and returns the device name if available.
def checkGPU(device):
    if device == 'cuda':
        if torch.cuda.is_available():
            messegeDividingLine('Using GPU for training! (つ´ω`)つ')
            return 'cuda'
        else:
            messegeDividingLine('No GPU! 。･ﾟ･(つд`)･ﾟ･')
            return 'cpu'
    elif device == 'cpu':
        if torch.cuda.is_available():
            messegeDividingLine('Using cpu for training! (つ´ω`)つ')
    else:
        raise KeyError(f'Wrong indication for the device: {device}')

# This function sets the seed for both PyTorch and numpy.
def setSeed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

# This function selects the dataset specified by dataset_name.
def datasetSelector(dataset_name = 'MNIST', save_dir = 'MNIST/', num_workers = 1, pin_memory = True, batch_size = 64, shuffle = False):
    if dataset_name == 'MNIST':
        from dataset import mnistDataset
        return mnistDataset(save_dir = save_dir, num_workers = num_workers, pin_memory = pin_memory, batch_size = batch_size, shuffle = shuffle)

# This function selects the model architecture specified by model_name.
def modelSelector(model_name, device, show_model = True):
    if model_name == 'CustomMLPNetwork':
        from model import CustomMLPNetwork
        model = CustomMLPNetwork().to(device)
    if model_name == 'CustomCNN':
        from model import CustomCNN
        model = CustomCNN().to(device)

    # prints the model architecture if show_model is True.
    if show_model:
        print(model)

    return model

# This function selects the optimizer specified by optimizer_name and returns it with the specified learning rate.
def optimizerSelector(optimizer_name, params, lr):
    if optimizer_name == 'Adam':
        from torch.optim import Adam
        return Adam(params = params, lr = lr)

# This function selects the loss function specified by loss_name and returns it.
def lossSelector(loss_name):
    if loss_name == 'CrossEntropyLoss':
        from torch.nn import CrossEntropyLoss
        return CrossEntropyLoss()

# This function saves the model checkpoint at the specified directory,
# which saves the model state dictionary, optimizer state dictionary, epoch, loss, and start date.
def saveModel(model, save_dir, model_name, start_date, epoch, loss, optimizer, train_loss_curve, train_acc_curve, val_loss_curve, val_acc_curve):
    checkpoint_dir = save_dir + model_name + start_date
    mkdir(checkpoint_dir)
    checkpoint_dir = checkpoint_dir + f'/epoch_{epoch}.pt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'date': start_date,
        'train_loss_curve': train_loss_curve,
        'train_acc_curve': train_acc_curve,
        'val_loss_curve': val_loss_curve,
        'val_acc_curve': val_acc_curve,
        }, checkpoint_dir)

    print(f' - Checkpoint saved! {checkpoint_dir} - ')

# This function loads the model checkpoint specified by load_dir,
# which returns the loaded model, optimizer, epoch, loss, and start date.
def loadModel(load_dir, model, optimizer, epoch, date, loss, train_loss_curve, train_acc_curve, val_loss_curve, val_acc_curve):
    checkpoint = torch.load(load_dir)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    date = checkpoint['date']
    train_loss_curve = checkpoint['train_loss_curve']
    train_acc_curve = checkpoint['train_acc_curve']
    val_loss_curve = checkpoint['val_loss_curve']
    val_acc_curve = checkpoint['val_acc_curve']
    singleDividingLine(f' - Keep training from: {date[1:]}, epoch: {epoch}, loss: {loss.item()} -')
    print(loss, epoch)
    return model, optimizer, epoch, loss, date, train_loss_curve, train_acc_curve, val_loss_curve, val_acc_curve

# This function initializes the start_time, start_date, start_epoch, and loss.
def initTrain():
    start_time = time.time()
    start_date = getCurrentTime()
    start_epoch = 0
    loss = 0
    train_accuracy = 0
    val_loss = 0
    val_accuracy = 0
    return start_time, start_date, start_epoch, loss, train_accuracy, val_loss, val_accuracy

# This function initializes the training and validation curve lists.
def initCurve():
    train_loss_curve = []
    train_acc_curve = []
    val_loss_curve = []
    val_acc_curve = []
    return train_loss_curve, train_acc_curve, val_loss_curve, val_acc_curve

# This function appends the loss and accuracy values to the respective curve lists.
def curveAppend(validation, train_loss_curve, train_acc_curve, val_loss_curve, val_acc_curve, loss, train_accuracy, val_loss, val_accuracy):
    train_loss_curve.append(loss)
    train_acc_curve.append(train_accuracy)
    if validation:
        val_loss_curve.append(val_loss)
        val_acc_curve.append(val_accuracy)

# This function plots the training and validation curves for both loss and accuracy. It saves the plot at the specified directory.
def plotCurve(train_loss_curve, train_acc_curve, val_loss_curve, val_acc_curve, validation, save_dir, model_name, start_date):
    img_dir = save_dir + model_name + start_date
    mkdir(img_dir)
    img_dir = img_dir + f'/curve.png'

    x_axis = [ _ for _ in  range(1, len(train_acc_curve) + 1)]

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(x_axis, train_loss_curve, label = 'train_loss')
    axs[1].plot(x_axis, train_acc_curve, label = 'train_acc')

    if validation:
        axs[0].plot(x_axis, val_loss_curve, label = 'val_loss')
        axs[1].plot(x_axis, val_acc_curve, label = 'val_acc')

    axs[0].legend()
    axs[1].legend()
    plt.savefig(img_dir)
    plt.show()

    print(f' - Figure saved! {img_dir} - ')