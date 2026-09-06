import numpy as np
from .core import Module


class MaxPool2D(Module):
  def __init__(self,
               pool_size: int,
               stride: int,
               prv_layer=None,
               next_layer=None):
    self.pool_size = pool_size
    self.stride = stride
    self.prv_layer = prv_layer
    self.next_layer = next_layer
    self.layer_output = None
    self.layer_delta_term = None
    self.mask = None
    self.x_shape = None

  def init(self) -> None:
    C, H, W = self.prv_layer.out_shape
    p, s = self.pool_size, self.stride
    self.out_H = (H - p) // s + 1
    self.out_W = (W - p) // s + 1
    self.out_shape = (C, self.out_H, self.out_W)

  def forward(self) -> None:
    x = self.prv_layer.layer_output  # (batch, C, H, W)
    batch, C, H, W = x.shape
    p, s = self.pool_size, self.stride
    self.x_shape = x.shape

    # Build a strided view of shape  (batch, C, out_H, out_W, p, p)
    # — zero-copy, no new allocation for the data
    shape = (batch, C, self.out_H, self.out_W, p, p)
    st = x.strides
    strides = (st[0], st[1], s * st[2], s * st[3], st[2], st[3])
    windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    out = windows.max(axis=(4, 5))  # (batch, C, out_H, out_W)
    # Mask: True where the maximum occurred (broadcast max back to pool dims)
    self.mask = (windows == out[:, :, :, :, None, None])
    self.layer_output = out

  def compute_delta_term(self, network, targets: np.ndarray) -> None:
    incoming = self.next_layer.layer_delta_term  # (batch, C, out_H, out_W)
    p, s = self.pool_size, self.stride

    d_input = np.zeros(self.x_shape, dtype=incoming.dtype)

    # Normalise mask so tied maxima share the gradient
    count = self.mask.sum(axis=(4, 5), keepdims=True).clip(min=1)
    d_windows = (incoming[:, :, :, :, None, None] * (self.mask / count)
                 )  # (batch, C, out_H, out_W, p, p)

    # Scatter back to spatial grid
    for oh in range(self.out_H):
      for ow in range(self.out_W):
        d_input[:, :, oh * s:oh * s + p,
                ow * s:ow * s + p] += d_windows[:, :, oh, ow]

    self.layer_delta_term = d_input

  def update_weights(self, lr: float) -> None:
    pass  # no learnable parameters

class Flatten(Module):

  def __init__(self, prv_layer=None, next_layer=None):
    self.prv_layer = prv_layer
    self.next_layer = next_layer
    self.layer_output = None
    self.layer_delta_term = None
    self.conv_shape = None  # saved during forward for backward reshape

  def init(self) -> None:
    flat_size = int(np.prod(self.prv_layer.out_shape))
    self.out_shape = (flat_size, )
    self.n_neuron = flat_size  # alias used by DenseLayer.init()

  def forward(self) -> None:
    x = self.prv_layer.layer_output  # (batch, C, H, W)
    self.conv_shape = x.shape
    self.layer_output = x.reshape(x.shape[0], -1)

  def compute_delta_term(self, network, targets: np.ndarray) -> None:
    incoming = self.next_layer.layer_delta_term
    if hasattr(self.next_layer, 'weights'):
      incoming = incoming @ self.next_layer.weights
    self.layer_delta_term = incoming.reshape(self.conv_shape)

  def update_weights(self, lr: float) -> None:
    pass