import numpy as np
from typing import Callable
from .core import Module


class Dense(Module):
  def __init__(self,
               n_neuron: int,
               act_func: Callable,
               use_dropout: bool = False,
               drop_rate: float = 0.3,
               weights_initialization: str = "He"):
    super().__init__()
    self.n_neuron = n_neuron
    self.act_func = act_func
    self.use_dropout = use_dropout
    self.drop_rate = drop_rate
    self.w_init_method = weights_initialization
    self.network = None

  def init(self):
    # Accepts Flatten (out_shape) or Dense (n_neuron)
    if hasattr(self.prv_layer, 'out_shape'):
      n_in = int(np.prod(self.prv_layer.out_shape))
    elif hasattr(self.prv_layer, 'n_neuron'):
      n_in = self.prv_layer.n_neuron
    else:
      n_in = self.prv_layer.layer_output.shape[-1]

    _temp = np.random.default_rng().standard_normal(size=(self.n_neuron, n_in))
    if self.w_init_method == "He":
      self.weights = (_temp * np.sqrt(2 / n_in)).astype(np.float32)
    else:
      self.weights = (_temp * np.sqrt(1 / n_in)).astype(np.float32)

    self.biases = np.zeros(self.n_neuron, dtype=np.float32)
    self.out_shape = (self.n_neuron, )
    self.Z = None

  def forward(self, predict_input: np.ndarray = None) -> np.ndarray:
    inp = predict_input if predict_input is not None else self.prv_layer.layer_output
    self.Z = inp @ self.weights.T + self.biases
    out = self.act_func(self.Z)

    training = self.network.training if hasattr(self.network,
                                                'training') else True
    if self.use_dropout and training:
      self.drop_mask = (np.random.default_rng().random(out.shape)
                        > self.drop_rate).astype(np.float32)
      out = (out * self.drop_mask) / (1.0 - self.drop_rate)

    self.layer_output = out
    return self.layer_output

  def compute_delta_term(self, network, targets: np.ndarray) -> None:
    if self.next_layer is None:
      if getattr(self.act_func, '__name__', '') == "softmax":
        grad_loss = network.compute_loss(derived=True,
                                         targets=targets)[:, None, :]
        soft_jacobian = self.act_func(self.Z, derived=True)
        delta = (grad_loss @ soft_jacobian).reshape(grad_loss.shape[0],
                                                    grad_loss.shape[-1])
      else:
        grad_loss = network.compute_loss(derived=True, targets=targets)
        delta = self.act_func(self.Z, derived=True) * grad_loss
    else:
      incoming = self.next_layer.layer_delta_term
      if hasattr(self.next_layer, 'weights'):
        incoming = incoming @ self.next_layer.weights
      elif hasattr(self.next_layer, 'conv_shape'):  # From flatten backward
        incoming = incoming

      delta = self.act_func(self.Z, derived=True) * incoming

    if self.use_dropout and (hasattr(self.network, 'training')
                             and self.network.training):
      delta = (delta * self.drop_mask) / (1.0 - self.drop_rate)

    self.layer_delta_term = delta

    # Compute gradients
    inp = self.prv_layer.layer_output
    self.dW = self.layer_delta_term.T @ inp
    self.db = np.sum(self.layer_delta_term, axis=0)

  def update_weights(self, lr: float) -> None:
    self.weights -= lr * self.dW
    self.biases -= lr * self.db


class InputLayer(Module):
  def __init__(self, layer_input: np.ndarray | None = None, input_shape: tuple | None = None):
    super().__init__()
    self.layer_input = layer_input

    if input_shape is None:
      self.out_shape = layer_input.shape[1:]
      self.n_neuron = layer_input.shape[1] if len(layer_input.shape) == 2 else np.prod(layer_input.shape[1:])
    else:
      self.out_shape = input_shape[1:]
      self.n_neuron = input_shape[1] if len(input_shape) == 2 else np.prod(input_shape[1:])

  def init(self):
    pass

  def forward(self, predict_input: np.ndarray = None) -> np.ndarray:
    self.layer_output = predict_input if predict_input is not None else self.layer_input
    return self.layer_output

  def compute_delta_term(self, network, targets: np.ndarray) -> None:
    pass

  def update_weights(self, lr: float) -> None:
    pass
