import time
from tools import *
from config import *
from tqdm import tqdm
from argparse import ArgumentParser


def train(model, loss_function, optimizer, epochs, device, train_data, validation_data, validation, save_model, load_model, save_dir, load_dir, model_name):
    start_time, start_date, start_epoch, loss, train_accuracy, val_loss, val_accuracy = initTrain()
    train_loss_curve, train_acc_curve, val_loss_curve, val_acc_curve = initCurve()

    if load_model:
        model, optimizer, start_epoch, loss, start_date, train_loss_curve, train_acc_curve, val_loss_curve, val_acc_curve \
                                                        = loadModel(
                                                                        load_dir = load_dir,
                                                                        model = model,
                                                                        optimizer = optimizer,
                                                                        epoch = start_epoch,
                                                                        date = start_date,
                                                                        loss = loss,
                                                                        train_loss_curve = train_loss_curve,
                                                                        train_acc_curve = train_acc_curve,
                                                                        val_loss_curve = val_loss_curve,
                                                                        val_acc_curve = val_acc_curve,
                                                                    )

    for epoch in range(start_epoch + 1, epochs + 1):
        messegeDividingLine(F' - Epoch {epoch} -')
        with tqdm(train_data) as train_bar:
            for data, label in train_bar:

                train_bar.set_description(f' - Training Epoch {epoch}')
                model.train()

                data = data.to(device)
                label = label.to(device)

                optimizer.zero_grad()
                outputs = model(data)

                # return index of the maximum value.
                predictions = outputs.argmax(dim = 1, keepdim = True).squeeze()
                correct = (predictions == label).sum().item()

                train_accuracy = 100. * (correct / float(data.shape[0]))

                loss = loss_function(outputs, label)
                loss.backward()
                optimizer.step()

                train_bar.set_postfix(train_loss = loss.item(), train_acc = train_accuracy)

        if validation:
            with tqdm(validation_data) as val_bar:
                for val_data, val_label in val_bar:

                    val_bar.set_description(f' - Validation Epoch {epoch}')
                    model.eval()

                    with torch.no_grad():

                        val_data = val_data.to(device)
                        val_label = val_label.to(device)
                        val_outputs = model(val_data)

                        val_predictions = val_outputs.argmax(dim = 1, keepdim = True).squeeze()
                        val_correct = (val_predictions == val_label).sum().item()
                        val_accuracy = 100. * (val_correct / float(val_data.shape[0]))
                        val_loss = loss_function(val_outputs, val_label)
                        val_loss = val_loss.item()

                        val_bar.set_postfix(val_loss = val_loss, val_acc = val_accuracy)

        curveAppend(
                        validation,
                        train_loss_curve,
                        train_acc_curve,
                        val_loss_curve,
                        val_acc_curve,
                        loss.item(),
                        train_accuracy,
                        val_loss,
                        val_accuracy,
                    )



        if save_model:
            saveModel(model, save_dir, model_name, start_date, epoch, loss, optimizer,
                                        train_loss_curve, train_acc_curve, val_loss_curve, val_acc_curve)

    plotCurve(train_loss_curve, train_acc_curve, val_loss_curve, val_acc_curve, validation, save_dir, model_name, start_date)
    time_consume = time.time() - start_time
    messegeDividingLine(f' - Training Finished - \n - Time Consuming: {time_consume:.3f} sec. - ')


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--DEVICE', help='cpu / cuda', type = str, default = DEVICE)
    parser.add_argument('--MODEL', help='choose a model', type = str, default = MODEL)
    parser.add_argument('--OPTIMIZER', help='choose an optimizer', type = str, default = OPTIMIZER)
    parser.add_argument('--LOSS', help='choose a loss', type = str, default = LOSS)
    parser.add_argument('--SAVE_MODEL', help='save the model', action= 'store_true', default = SAVE_MODEL)
    parser.add_argument('--LOAD_MODEL', help='load the model',  action= 'store_true', default = LOAD_MODEL)
    parser.add_argument('--NUM_EPOCHS', help='total epochs for training', type = int, default = NUM_EPOCHS)
    parser.add_argument('--LEARNING_RATE', help='learning rate for tuning parameter', type = float, default = LEARNING_RATE)
    parser.add_argument('--DATA_DIR', help='direction to save / access the data', type = str, default = DATA_DIR)
    parser.add_argument('--DATASET_NAME', help='name of the dataset', type = str, default = DATASET_NAME)
    parser.add_argument('--PIN_MEMORY', help='-', type = bool, default = PIN_MEMORY)
    parser.add_argument('--WORKERS', help='number of cpu cores', type = int, default = NUM_WORKERS)
    parser.add_argument('--SHUFFLE_DATA', help='shffle the training data', type = bool, default = SHUFFLE_DATA)
    parser.add_argument('--BATCH_SIZE', help='determine the batch size for training', type = int, default = BATCH_SIZE)
    parser.add_argument('--VALIDATION', help='validate the training process', type = bool, default = VALIDATION)
    parser.add_argument('--SHOW_MODEL', help='briefly display the applying model', type = bool, default = SHOW_MODEL)
    parser.add_argument('--DISPLAY_SAMPLE', help='visualize the sample data', type = int, default = DISPLAY_SAMPLE)
    parser.add_argument('--SAVE_CHECKPOINT_DIR', help='direction to save checkpoints', type = str, default = SAVE_CHECKPOINT_DIR)
    parser.add_argument('--LOAD_CHECKPOINT_DIR', help='direction to load the checkpoint', type = str, default = LOAD_CHECKPOINT_DIR)

    args = parser.parse_args()

    initMessege(args)

    model = modelSelector(model_name = args.MODEL, device = args.DEVICE, show_model = args.SHOW_MODEL)
    loss_function = lossSelector(loss_name = args.LOSS)
    optimizer = optimizerSelector(optimizer_name = args.OPTIMIZER, params = model.parameters(), lr = args.LEARNING_RATE)

    train_data, val_data = datasetSelector(
                                            dataset_name = args.DATASET_NAME,
                                            save_dir = args.DATA_DIR,
                                            num_workers = args.WORKERS,
                                            pin_memory = args.PIN_MEMORY,
                                            batch_size = args.BATCH_SIZE,
                                            shuffle = args.SHUFFLE_DATA,
                                        )

    if args.DISPLAY_SAMPLE != 0:
        displaySample(data = train_data, num = 3)

    train(
            model = model,
            loss_function = loss_function,
            optimizer = optimizer,
            epochs = args.NUM_EPOCHS,
            device = args.DEVICE,
            train_data = train_data,
            validation_data = val_data,
            validation = args.VALIDATION,
            save_model = args.SAVE_MODEL,
            load_model = args.LOAD_MODEL,
            save_dir = args.SAVE_CHECKPOINT_DIR,
            load_dir = args.LOAD_CHECKPOINT_DIR,
            model_name = args.MODEL,
        )