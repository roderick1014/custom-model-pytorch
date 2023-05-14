# **custom-model-pytorch✅**
##### **This is a sample code for deep learning based on PyTorch, suitable for beginners to practice and learn.It provides convenience when switching between models, loss functions, and optimizers. Additionally, it incorporates features such as tqdm and argparse. It also includes functionality to continue training from a saved training state.**
---

<!-- ## ArgumentParser -->
**🔺Note: If you are not familiar with `ArgumentParser`, that's okay. You can easily set the parameters mentioned above in the `config.py` file.🙂**

The code snippet you provided defines an argument parser using the ArgumentParser class from the argparse module. Here are the explanations of the arguments being added in **`train.py`**:
  ```python
  parser = ArgumentParser()
  parser.add_argument('--DEVICE', help='cpu / cuda', type=str, default=DEVICE)
  parser.add_argument('--MODEL', help='choose a model', type=str, default=MODEL)
  parser.add_argument('--OPTIMIZER', help='choose an optimizer', type=str, default=OPTIMIZER)
  parser.add_argument('--LOSS', help='choose a loss', type=str, default=LOSS)
  parser.add_argument('--SAVE_MODEL', help='save the model', action='store_true', default=SAVE_MODEL)
  parser.add_argument('--LOAD_MODEL', help='load the model', action='store_true', default=LOAD_MODEL)
  parser.add_argument('--NUM_EPOCHS', help='total epochs for training', type=int, default=NUM_EPOCHS)
  parser.add_argument('--LEARNING_RATE', help='learning rate for tuning parameter', type=float, default=LEARNING_RATE)
  parser.add_argument('--DATA_DIR', help='direction to save / access the data', type=str, default=DATA_DIR)
  parser.add_argument('--PIN_MEMORY', help='-', type=bool, default=PIN_MEMORY)
  parser.add_argument('--WORKERS', help='number of CPU cores', type=int, default=NUM_WORKERS)
  parser.add_argument('--BATCH_SIZE', help='determine the batch size for training', type=int, default=BATCH_SIZE)
  parser.add_argument('--VALIDATION', help='validate the training process', type=bool, default=VALIDATION)
  parser.add_argument('--SHOW_MODEL', help='briefly display the applying model', type=bool, default=SHOW_MODEL)
  parser.add_argument('--DISPLAY_SAMPLE', help='visualize the sample data', type=int, default=DISPLAY_SAMPLE)
  parser.add_argument('--SAVE_CHECKPOINT_DIR', help='direction to save checkpoints', type=str, default=SAVE_CHECKPOINT_DIR)
  parser.add_argument('--LOAD_CHECKPOINT_DIR', help='direction to load the checkpoint', type=str, default=LOAD_CHECKPOINT_DIR)

  args = parser.parse_args()
  ```
 **The code above creates an argument parser using the ArgumentParser class. It sets up various command-line arguments that can be passed when running the script. Here's a summary of each argument:**

* **`--DEVICE`**: Specifies the device to use (cpu / cuda).
* **`--MODEL`**: Allows choosing a specific model.
* **`--OPTIMIZER`**: Allows choosing an optimizer.
* **`--LOSS`**: Allows choosing a specific loss function.
* **`--SAVE_MODEL`**: Enables saving the model.
* **`--LOAD_MODEL`**: Enables loading a pre-trained model.
* **`--NUM_EPOCHS`**: Sets the total number of training epochs.
* **`--LEARNING_RATE`**: Sets the learning rate for parameter tuning.
* **`--DATA_DIR`**: Specifies the directory to save/access the data.
* **`--PIN_MEMORY`**: A boolean flag (true/false) to enable pinned memory.
* **`--WORKERS`**: Sets the number of CPU cores to use.
* **`--BATCH_SIZE`**: Sets the batch size for training.
* **`--VALIDATION`**: Enables validation during the training process.
* **`--SHOW_MODEL`**: Displays a brief summary of the applied model.
* **`--DISPLAY_SAMPLE`**: Specifies the number of sample data to visualize.
* **`--SAVE_CHECKPOINT_DIR`**: Specifies the directory to save checkpoints.
* **`--LOAD_CHECKPOINT_DIR`**: Specifies the directory to load a checkpoint from.
The args variable stores the parsed command-line arguments. These arguments provide flexibility to modify the behavior of the script based on user input.
