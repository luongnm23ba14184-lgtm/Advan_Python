import numpy as np

class ActivationFunction:

  @staticmethod
  def identity(Z: np.ndarray, derived: bool = False) -> np.ndarray:
    if derived:
      return np.ones_like(Z)
    else:
      return Z

  @staticmethod
  def ReLU(Z: np.ndarray, derived: bool = False) -> np.ndarray:
    if derived:
      return (Z > 0).astype(float)
    else:
      return np.maximum(0, Z)

  @staticmethod
  def sigmoid(Z: np.ndarray, derived: bool = False) -> np.ndarray:
    # np.clip to prevent overflow
    Z_clipped = np.clip(Z, -500, 500)
    sigmoid = 1 / (1 + np.exp(-Z_clipped))
    if derived:
      return sigmoid * (1 - sigmoid)
    else:
      return sigmoid

  @staticmethod
  def softmax(Z: np.ndarray, derived: bool = False) -> np.ndarray:
    Z = Z - np.max(Z, axis=1, keepdims=True)
    numerator = np.exp(Z)
    denominator = np.sum(numerator, axis=1, keepdims=True)
    S = numerator / denominator
    if derived:
      s_reshaped = S[:, :, None]
      diag_part = s_reshaped * np.eye(S.shape[1])[None, :, :]
      outer_part = s_reshaped * s_reshaped.transpose(0, 2, 1)
      return diag_part - outer_part
    else:
      return S

  @staticmethod
  def Tanh(Z: np.ndarray, derived=False) -> np.ndarray:
    tanh = np.tanh(Z)
    if derived:
      return 1 - tanh**2
    else:
      return tanh


class LossFunction:

  @staticmethod
  def cc_loss(predicted_values: np.ndarray,
              targets: np.ndarray,
              derived: bool = False):
    batch = predicted_values.shape[0]
    if derived:
      return (-1 / batch) * targets / (predicted_values + 1e-15)
    else:
      class_indices = np.argmax(targets, axis=1)
      return (-1 / batch) * np.sum(
          np.log(predicted_values[np.arange(batch), class_indices] + 1e-15))

  @staticmethod
  def MSE(predicted_values: np.ndarray,
          targets: np.ndarray,
          derived: bool = False):
    batch = predicted_values.shape[0]
    if derived:
      return (1 / batch) * (predicted_values - targets)
    else:
      return (1 / (2 * batch)) * np.sum((predicted_values - targets)**2)
