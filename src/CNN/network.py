import numpy as np
from typing import Callable

class Network:
  def __init__(self,
               layers: list,
               loss_func: Callable,
               training_set: tuple[np.ndarray, np.ndarray] | None = None,
               batch: int | None = None,
               learning_rate: float = 0.01,
               lr_decay: float = 1.0,
               epsilon: float | None = 0.0001,
               epoch_limit: int | None = 10,
               test_set: tuple[np.ndarray, np.ndarray] | None = None,
               iteration_event_trigger: int | None = -1,
               eval_every: int = 0,
               exponential_moving_average: float = 0.2,
               augment_fn: Callable | None = None):

    # If training set is not specified, user tends to load pre-trained weights
    if training_set:
      self.x_train = training_set[0]
      self.y_train = training_set[1]
      self.batch = batch if batch is not None else self.x_train.shape[0]

    self.beta = exponential_moving_average
    self.augment_fn = augment_fn
    self.network_layers = layers
    self.loss_func = loss_func
    self.learning_rate = learning_rate
    self.lr_decay = lr_decay
    self.epsilon = epsilon
    self.epoch_limit = epoch_limit
    self.iet = iteration_event_trigger
    self.eval_every = eval_every

    self.test_set_exist = test_set is not None
    if self.test_set_exist:
      self.x_test = test_set[0]
      self.y_test = test_set[1]

    self.network_loss = 0
    self.iterations = 0
    self.epoch = 0
    self.training = True  # flag for Dropout/BatchNorm

    self._init_network()

  def _init_network(self) -> None:
    # Link layers forward and backward
    indexes = len(self.network_layers)
    for i in range(indexes):
      if i > 0:
        self.network_layers[i].prv_layer = self.network_layers[i - 1]
      if i < indexes - 1:
        self.network_layers[i].next_layer = self.network_layers[i + 1]

      # inject network reference
      self.network_layers[i].network = self
      # Call init on each layer
      self.network_layers[i].init()

    print("--- Neural network is initialized successfully ---")

  def compute_loss(self,
                   targets: np.ndarray,
                   derived: bool = False) -> np.ndarray:
    predicted_vals = self.network_layers[-1].layer_output
    loss = self.loss_func(predicted_values=predicted_vals,
                          targets=targets,
                          derived=derived)
    if not derived:
      self.network_loss = loss
    return loss

  def _forward_propagation(self, predict_input: np.ndarray = None) -> None:
    self.network_layers[0].forward(predict_input)
    for layer in self.network_layers[1:]:
      layer.forward()

  def _backward_propagation(self, targets: np.ndarray, current_lr: float = None) -> None:
    if current_lr is None:
      current_lr = self.learning_rate
      
    for layer in self.network_layers[-1:0:-1]:
      layer.compute_delta_term(network=self, targets=targets)

    # Update weights (can be replaced by an external Optimizer class if needed)
    for layer in self.network_layers[1:]:
      layer.update_weights(current_lr)

  def fit_model(self) -> None:
    if self.eval_every == 0 and not self.test_set_exist:
      pass  # no test set; skip end-of-training eval silently
    print("--- Start training ---")
    self.training = True
    n_samples = self.x_train.shape[0]
    smoothed_loss = None
    prv_smoothed_loss = None
    beta = self.beta

    stop = False
    while not stop:
      if self.epoch == self.epoch_limit:
        stop = True
        break

      indices = np.random.default_rng().permutation(n_samples)
      self.epoch += 1
      current_lr = self.learning_rate * (self.lr_decay ** (self.epoch - 1))

      for i in range(0, n_samples, self.batch):
        batch_indices = indices[i:i + self.batch]

        x_batch = self.x_train[batch_indices]
        if self.augment_fn is not None:
          x_batch = self.augment_fn(x_batch)

        self._forward_propagation(predict_input=x_batch)
        new_loss = self.compute_loss(targets=self.y_train[batch_indices])

        self._backward_propagation(targets=self.y_train[batch_indices], current_lr=current_lr)

        if smoothed_loss is None:
          smoothed_loss = new_loss
        else:
          smoothed_loss = beta * smoothed_loss + (1 - beta) * new_loss

        if prv_smoothed_loss is not None:
          smoothed_loss_diff = abs(prv_smoothed_loss -
                                   smoothed_loss) / smoothed_loss
          if smoothed_loss_diff < self.epsilon:
            stop = True
            break

        if self.iet > 0 and self.iterations % self.iet == 0:
          print(
              f"Updates: #{self.iterations} | Loss: {new_loss:.6f} | Epoch: #{self.epoch} | clr = {current_lr:.4f}"
          )
        self.iterations += 1

        prv_smoothed_loss = smoothed_loss

      # Per-epoch evaluation on test set
      if self.test_set_exist and self.eval_every > 0 and self.epoch % self.eval_every == 0:
        acc = self.evaluate()
        self.training = True  # restore training mode after evaluate()
        if acc is not None:
          print(f"Epoch #{self.epoch} | Test evaluation: {acc:.2f}%")

  def predict(self, input: np.ndarray) -> np.ndarray:
    self.training = False
    
    # Process in batches to save memory (prevents massive im2col allocations) when passing 10k images into model
    batch_size = self.batch if hasattr(self, 'batch') and self.batch is not None else 256
    results = []
    n_samples = input.shape[0]
    
    for i in range(0, n_samples, batch_size):
      batch_input = input[i:i + batch_size]
      self._forward_propagation(predict_input=batch_input)
      results.append(self.network_layers[-1].layer_output)
      
    return np.concatenate(results, axis=0)

  def evaluate(self) -> float:
    self.training = False
    model = getattr(self.loss_func, '__name__', 'unknown')

    if model == "cc_loss":
      act_classes = np.argmax(self.y_test, axis=1)
      predictions = self.predict(input=self.x_test)
      pred_classes = np.argmax(predictions, axis=1)
      precision = np.mean(pred_classes == act_classes) * 100
      return precision
    elif model == "MSE":
      if not self.test_set_exist:
        print("--- Warming, model does not have test set for evaluation ---")
        return None
      else:
        act_mean = np.mean(self.y_test)
        sst = np.sum((self.y_test - act_mean)**2)
        predictions = self.predict(input=self.x_test)
        ssr = np.sum((self.y_test - predictions)**2)
        return (1 - ssr / sst) * 100
    return None

  def save_weights(self, path: str, accuracy: float = None) -> None:
    arrays = {}
    for idx, layer in enumerate(self.network_layers[1:], 1):
      if hasattr(layer, 'weights') and layer.weights is not None:
        arrays[f"l{idx}_weights"] = layer.weights
        arrays[f"l{idx}_biases"] = layer.biases
      if hasattr(layer, 'kernels') and layer.kernels is not None:
        arrays[f"l{idx}_kernels"] = layer.kernels
        arrays[f"l{idx}_biases"] = layer.biases
      if hasattr(layer, 'use_bn') and layer.use_bn:
        arrays[f"l{idx}_bn_gamma"] = layer.bn_gamma
        arrays[f"l{idx}_bn_beta"] = layer.bn_beta
        arrays[f"l{idx}_bn_run_mean"] = layer.bn_run_mean
        arrays[f"l{idx}_bn_run_var"] = layer.bn_run_var
    if accuracy is not None:
      arrays["accuracy"] = np.float32(accuracy)
    np.savez(path, **arrays)
    acc_str = f"  |  accuracy: {accuracy:.2f}%" if accuracy is not None else ""
    print(f"Weights saved -> {path}.npz{acc_str}")

  def load_weights(self, path: str) -> float | None:
    data = np.load(path if path.endswith('.npz') else path + '.npz')
    
    # Auto-detect legacy indexing (nn used 0-based, cnn used 1-based)
    start_idx = 0 if any(k.startswith('l0_') for k in data.files) else 1

    for idx, layer in enumerate(self.network_layers[1:], start_idx):
      if f"l{idx}_weights" in data:
        layer.weights = data[f"l{idx}_weights"]
        layer.biases = data[f"l{idx}_biases"]
      if f"l{idx}_kernels" in data:
        layer.kernels = data[f"l{idx}_kernels"]
        layer.biases = data[f"l{idx}_biases"]
      if f"l{idx}_bn_gamma" in data:
        layer.bn_gamma = data[f"l{idx}_bn_gamma"]
        layer.bn_beta = data[f"l{idx}_bn_beta"]
        layer.bn_run_mean = data[f"l{idx}_bn_run_mean"]
        layer.bn_run_var = data[f"l{idx}_bn_run_var"]

    accuracy = float(data["accuracy"]) if "accuracy" in data else None
    acc_str = f"  |  accuracy: {accuracy:.2f}%" if accuracy is not None else ""
    print(f"Weights loaded <- {path}{acc_str}")
    return accuracy
