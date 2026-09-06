import numpy as np
import pickle
import time
import os

from CNN.network import Network
from CNN.conv import Conv2D
from CNN.pooling import MaxPool2D
from CNN.pooling import Flatten
from CNN.linear import Dense, InputLayer
from CNN.functional import ActivationFunction, LossFunction

# -------------``
# 1. Load data
# -------------

WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), '..',  'weights', "cnn_cifar10_weights.npz")

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
data_dir = os.path.join(os.path.dirname(__file__), "..", "dataset", "cifar-10-batches-py")
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
  Dense(512, act_func=act.ReLU, use_dropout=True, drop_rate=0.3),
  Dense(10,  act_func=act.softmax, use_dropout=False, drop_rate=0.0)
]

epoch_limit = 10
gamma = 0.01**(1/epoch_limit)

print(f'--- Learning rate deay = {gamma:.5f} for epoch limit of {epoch_limit} --')

model = Network(
  layers=layers,
  training_set=(x_train, y_train),
  test_set=(x_test, y_test),
  loss_func=LossFunction.cc_loss,
  batch=256,
  learning_rate=0.05, # Can start a bit lower for CIFAR
  lr_decay=gamma,
  epsilon=1e-4,
  epoch_limit=epoch_limit, # Needs more epochs than MNIST
  iteration_event_trigger=1,
  eval_every=1,
  # exponential_moving_average= 0.2
)

# -------------
# 3. Train
# -------------

print("\nTraining CIFAR-10 (this will take longer than MNIST!) ...\n")
t0 = time.time()
model.fit_model()
elapsed = time.time() - t0
print(f"\nTraining done in {elapsed / 60:.1f} min")

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

# ~deep-learning> uv run .\training\cnn_train_cifar10.py
# Loading CIFAR-10 ...
#   x_train: (50000, 3, 32, 32)  x_test: (10000, 3, 32, 32)
# --- Neural network is initialized successfully ---

# Training CIFAR-10 (this will take longer than MNIST!) ...

# --- Start training ---
# Updates: #0 | Loss: 4.400631 | Epoch: #1
# Updates: #20 | Loss: 2.007273 | Epoch: #1
# Updates: #40 | Loss: 1.871899 | Epoch: #1
# Updates: #60 | Loss: 1.651663 | Epoch: #1
# Updates: #80 | Loss: 1.692833 | Epoch: #1
# Updates: #100 | Loss: 1.734249 | Epoch: #1
# Updates: #120 | Loss: 1.564528 | Epoch: #1
# Updates: #140 | Loss: 1.525574 | Epoch: #1
# Updates: #160 | Loss: 1.490286 | Epoch: #1
# Updates: #180 | Loss: 1.423811 | Epoch: #1
# Epoch #1 | Test evaluation: 49.13%
# Updates: #200 | Loss: 1.400773 | Epoch: #2
# Updates: #220 | Loss: 1.380766 | Epoch: #2
# Updates: #240 | Loss: 1.404663 | Epoch: #2
# Updates: #260 | Loss: 1.409914 | Epoch: #2
# Updates: #280 | Loss: 1.427707 | Epoch: #2
# Updates: #300 | Loss: 1.290929 | Epoch: #2
# Updates: #320 | Loss: 1.283340 | Epoch: #2
# Updates: #340 | Loss: 1.490432 | Epoch: #2
# Updates: #360 | Loss: 1.365967 | Epoch: #2
# Updates: #380 | Loss: 1.335225 | Epoch: #2
# Epoch #2 | Test evaluation: 54.36%
# Updates: #400 | Loss: 1.336786 | Epoch: #3
# Updates: #420 | Loss: 1.334505 | Epoch: #3
# Updates: #440 | Loss: 1.242628 | Epoch: #3
# Updates: #460 | Loss: 1.245814 | Epoch: #3
# Updates: #480 | Loss: 1.257568 | Epoch: #3
# Updates: #500 | Loss: 1.346426 | Epoch: #3
# Updates: #520 | Loss: 1.257514 | Epoch: #3
# Updates: #540 | Loss: 1.088264 | Epoch: #3
# Updates: #560 | Loss: 1.109687 | Epoch: #3
# Updates: #580 | Loss: 1.094513 | Epoch: #3
# Epoch #3 | Test evaluation: 53.05%
# Updates: #600 | Loss: 1.212468 | Epoch: #4
# Updates: #620 | Loss: 1.187761 | Epoch: #4
# Updates: #640 | Loss: 1.115483 | Epoch: #4
# Updates: #660 | Loss: 1.091724 | Epoch: #4
# Updates: #680 | Loss: 1.193729 | Epoch: #4
# Updates: #700 | Loss: 1.103901 | Epoch: #4
# Updates: #720 | Loss: 1.172821 | Epoch: #4
# Updates: #740 | Loss: 0.995148 | Epoch: #4
# Updates: #760 | Loss: 1.274538 | Epoch: #4
# Updates: #780 | Loss: 0.996562 | Epoch: #4
# Epoch #4 | Test evaluation: 57.35%
# Updates: #800 | Loss: 1.032185 | Epoch: #5
# Updates: #820 | Loss: 1.069117 | Epoch: #5
# Updates: #840 | Loss: 1.115569 | Epoch: #5
# Updates: #860 | Loss: 0.983694 | Epoch: #5
# Updates: #880 | Loss: 1.008473 | Epoch: #5
# Updates: #900 | Loss: 0.992074 | Epoch: #5
# Updates: #920 | Loss: 1.059741 | Epoch: #5
# Updates: #940 | Loss: 1.146259 | Epoch: #5
# Updates: #960 | Loss: 1.039901 | Epoch: #5
# Epoch #5 | Test evaluation: 52.84%

# Training done in 195.4 min
# Test accuracy: 52.84%
# Weights saved -> C:\Users\admin\projects\deep-learning\training\..\weights\cnn_cifar10_weights.npz.npz  |accuracy: 52.84%