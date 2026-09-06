import numpy as np


class Module:
  """
  Base class for all neural network layers.
  All subclasses should implement `forward` and `backward`.
  """
  def __init__(self):
    self.prv_layer = None
    self.next_layer = None
    self.layer_output = None
    self.layer_delta_term = None
    self.weights = None
    self.biases = None

  def init(self):
    """
    Initialize weights and biases. Called after layers are linked.
    """
    pass

  def forward(self, predict_input: np.ndarray = None) -> np.ndarray:
    """
    Forward pass. Computes `self.layer_output`.
    """
    raise NotImplementedError

  def compute_delta_term(self, network, targets: np.ndarray) -> None:
    """
    Backward pass. Computes `self.layer_delta_term` and potentially gradients.
    """
    raise NotImplementedError

  def update_weights(self, lr: float) -> None:
    """
    Update learnable parameters using computed gradients.
    """
    pass
