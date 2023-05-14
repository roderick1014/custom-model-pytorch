
DATA_DIR = 'MNIST/'
DEVICE = 'cuda'

SAVE_CHECKPOINT_DIR = "checkpoints/"
LOAD_CHECKPOINT_DIR = "checkpoints/CustomCNN_20230514_125822/epoch_2.pt"

SAVE_MODEL = True
LOAD_MODEL = False

LEARNING_RATE = 1e-4
DATASET_NAME = 'MNIST'
PIN_MEMORY = True
NUM_WORKERS = 0
BATCH_SIZE = 64
NUM_EPOCHS = 5
VALIDATION = False
MODEL = 'CustomMLPNetwork'  # CustomMLPNetwork, CustomCNN
OPTIMIZER = 'Adam'
LOSS = 'CrossEntropyLoss'
SHOW_MODEL = True
DISPLAY_SAMPLE = 0
SHUFFLE_DATA = True