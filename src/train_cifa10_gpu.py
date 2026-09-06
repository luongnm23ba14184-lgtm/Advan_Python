import numpy as np
import pickle
import time
import os

from src.CNN.cupy_cnn import Network, Conv2D, MaxPool2D, Flatten, Dense, InputLayer, ActivationFunction, LossFunction

# -------------
# 1. Load data
# -------------

WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), '..', 'weights', "cnn_cifar10_weights_GPU.npz")

CIFAR10_CLASSES = [
  "airplane", "automobile", "bird", "cat", "deer",
  "dog", "frog", "horse", "ship", "truck"
]

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10_data(data_dir):
    x_train_list, y_train_list = [], []
    for i in range(1, 6):
        batch = unpickle(os.path.join(data_dir, f"data_batch_{i}"))
        x_train_list.append(batch[b'data'])
        y_train_list += batch[b'labels']
    
    x_train_raw = np.concatenate(x_train_list)
    y_train_lbl = np.array(y_train_list)

    test_batch = unpickle(os.path.join(data_dir, "test_batch"))
    x_test_raw = test_batch[b'data']
    y_test_lbl = np.array(test_batch[b'labels'])

    return x_train_raw, y_train_lbl, x_test_raw, y_test_lbl

def onehot(labels, n=10):
  m = np.zeros((len(labels), n), dtype=np.float32)
  m[np.arange(len(labels)), labels] = 1
  return m

print("Loading CIFAR-10 ...")
data_dir = os.path.join(os.path.dirname(__file__), "dataset", "cifar-10-batches-py")
x_train_raw, y_train_lbl, x_test_raw, y_test_lbl = load_cifar10_data(data_dir)

# Shape: (N, 3, 32, 32), normalised to [0, 1]
x_train = (x_train_raw.reshape(-1, 3, 32, 32) / 255.0).astype(np.float32)
x_test  = (x_test_raw.reshape(-1, 3, 32, 32)  / 255.0).astype(np.float32)
y_train = onehot(y_train_lbl)
y_test  = onehot(y_test_lbl)
print(f"  x_train: {x_train.shape}  x_test: {x_test.shape}")

# -------------
# 2. Build model
# -------------

act = ActivationFunction

# We need a slightly deeper model for CIFAR-10 (color, more complex features)
layers = [
  InputLayer(x_train),
  Conv2D(32, 3, act_func=act.ReLU, stride=1, padding=1, use_bn=True),
  Conv2D(32, 3, act_func=act.ReLU, stride=1, padding=1, use_bn=True),
  MaxPool2D(2, 2),
  
  Conv2D(64, 3, act_func=act.ReLU, stride=1, padding=1, use_bn=True),
  Conv2D(64, 3, act_func=act.ReLU, stride=1, padding=1, use_bn=True),
  MaxPool2D(2, 2),
  
  Flatten(),
  Dense(512, act_func=act.ReLU, use_dropout=True, drop_rate=0),
  Dense(10,  act_func=act.softmax, use_dropout=False, drop_rate=0.0)
]

epoch_limit = 10
gamma = 0.01**(1/epoch_limit)

print(f'--- Learning rate decay = {gamma:.5f} for epoch limit of {epoch_limit} --')

model = Network(
  layers=layers,
  training_set=(x_train, y_train),
  test_set=(x_test, y_test),
  loss_func=LossFunction.cc_loss,
  batch=256,
  learning_rate=0.05,
  lr_decay=gamma,
  epsilon=1e-4,
  epoch_limit=epoch_limit,
  iteration_event_trigger=100,
  eval_every=1,
  exponential_moving_average= 0.2
)

# -------------
# 3. Train
# -------------

print("\nTraining CIFAR-10 on GPU (this should be significantly faster!) ...\n")
t0 = time.time()
model.fit_model()
elapsed = time.time() - t0
print(f"\nTraining done in {elapsed:.2f} seconds ({elapsed / 60:.1f} min)")

# -------------
# 4. Evaluate & save
# -------------

accuracy = model.evaluate()
print(f"Test accuracy: {accuracy:.2f}%")

res = '...'
while res != 'n':
  res = input("Want to save the pre-trained weights? -> 'y' for yes 'n' for no - ")
  if res == 'y':
    model.save_weights(WEIGHTS_FILE, accuracy=accuracy)
    break

# ~CNN> uv run .\train_cifa10_gpu.py
# Loading CIFAR-10 ...
#   x_train: (50000, 3, 32, 32)  x_test: (10000, 3, 32, 32)
# --- Learning rate decay = 0.39811 for epoch limit of 5 --
# --- Neural network is initialized successfully ---

# Training CIFAR-10 on GPU (this should be significantly faster!) ...

# --- Start training ---
# Updates: #0 | Loss: 3.040112 | Epoch: #1 | clr = 0.0500
# Updates: #100 | Loss: 1.625314 | Epoch: #1 | clr = 0.0500
# Epoch #1 | Test evaluation: 44.50%
# Updates: #200 | Loss: 1.396106 | Epoch: #2 | clr = 0.0199
# Updates: #300 | Loss: 1.134780 | Epoch: #2 | clr = 0.0199
# Epoch #2 | Test evaluation: 55.13%
# Updates: #400 | Loss: 1.175074 | Epoch: #3 | clr = 0.0079
# Updates: #500 | Loss: 1.127427 | Epoch: #3 | clr = 0.0079
# Epoch #3 | Test evaluation: 56.68%
# Updates: #600 | Loss: 1.158720 | Epoch: #4 | clr = 0.0032
# Updates: #700 | Loss: 1.274678 | Epoch: #4 | clr = 0.0032
# Epoch #4 | Test evaluation: 58.20%
# Updates: #800 | Loss: 1.029451 | Epoch: #5 | clr = 0.0013
# Updates: #900 | Loss: 1.144950 | Epoch: #5 | clr = 0.0013
# Epoch #5 | Test evaluation: 58.76%

# Training done in 1647.68 seconds (27.5 min)
# Test accuracy: 58.76%
# Want to save the pre-trained weights? -> 'y' for yes 'n' for no - y
# Weights saved -> C:\Users\admin\projects\CNN\cnn_cifar10_weights_GPU.npz.npz  |  accuracy: 58.76%