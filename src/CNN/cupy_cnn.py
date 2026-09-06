import cupy as cp
import cupyx
import numpy as np
from typing import Callable

# ==========================================
# 1. Core Module
# ==========================================
class Module:
    def __init__(self):
        self.prv_layer = None
        self.next_layer = None
        self.layer_output = None
        self.layer_delta_term = None
        self.weights = None
        self.biases = None

    def init(self):
        pass

    def forward(self, predict_input: cp.ndarray = None) -> cp.ndarray:
        raise NotImplementedError

    def compute_delta_term(self, network, targets: cp.ndarray) -> None:
        raise NotImplementedError

    def update_weights(self, lr: float) -> None:
        pass


# ==========================================
# 2. Functional (Activations & Losses)
# ==========================================
class ActivationFunction:
    @staticmethod
    def identity(Z: cp.ndarray, derived: bool = False) -> cp.ndarray:
        if derived:
            return cp.ones_like(Z)
        else:
            return Z

    @staticmethod
    def ReLU(Z: cp.ndarray, derived: bool = False) -> cp.ndarray:
        if derived:
            return (Z > 0).astype(float)
        else:
            return cp.maximum(0, Z)

    @staticmethod
    def sigmoid(Z: cp.ndarray, derived: bool = False) -> cp.ndarray:
        Z_clipped = cp.clip(Z, -500, 500)
        sigmoid = 1 / (1 + cp.exp(-Z_clipped))
        if derived:
            return sigmoid * (1 - sigmoid)
        else:
            return sigmoid

    @staticmethod
    def softmax(Z: cp.ndarray, derived: bool = False) -> cp.ndarray:
        Z = Z - cp.max(Z, axis=1, keepdims=True)
        numerator = cp.exp(Z)
        denominator = cp.sum(numerator, axis=1, keepdims=True)
        S = numerator / denominator
        if derived:
            s_reshaped = S[:, :, None]
            diag_part = s_reshaped * cp.eye(S.shape[1])[None, :, :]
            outer_part = s_reshaped * s_reshaped.transpose(0, 2, 1)
            return diag_part - outer_part
        else:
            return S

    @staticmethod
    def Tanh(Z: cp.ndarray, derived=False) -> cp.ndarray:
        tanh = cp.tanh(Z)
        if derived:
            return 1 - tanh**2
        else:
            return tanh

class LossFunction:
    @staticmethod
    def cc_loss(predicted_values: cp.ndarray, targets: cp.ndarray, derived: bool = False):
        batch = predicted_values.shape[0]
        if derived:
            return (-1 / batch) * targets / (predicted_values + 1e-15)
        else:
            class_indices = cp.argmax(targets, axis=1)
            return (-1 / batch) * cp.sum(cp.log(predicted_values[cp.arange(batch), class_indices] + 1e-15))

    @staticmethod
    def MSE(predicted_values: cp.ndarray, targets: cp.ndarray, derived: bool = False):
        batch = predicted_values.shape[0]
        if derived:
            return (1 / batch) * (predicted_values - targets)
        else:
            return (1 / (2 * batch)) * cp.sum((predicted_values - targets)**2)


# ==========================================
# 3. Convolutional Logic
# ==========================================
def _im2col_indices(C_in: int, kH: int, kW: int, out_H: int, out_W: int, stride: int):
    row_off = cp.repeat(cp.arange(kH), kW)
    row_off = cp.tile(row_off, C_in)
    row_out = stride * cp.arange(out_H)
    col_off = cp.tile(cp.arange(kW), kH)
    col_off = cp.tile(col_off, C_in)
    col_out = stride * cp.arange(out_W)
    k = cp.repeat(cp.arange(C_in), kH * kW)
    i = row_off[:, None] + row_out[None, :]
    j = col_off[:, None] + col_out[None, :]
    return k, i, j

def _im2col(x_pad: cp.ndarray, k: cp.ndarray, i: cp.ndarray, j: cp.ndarray, out_H: int, out_W: int) -> cp.ndarray:
    col = x_pad[:, k[:, None, None], i[:, :, None], j[:, None, :]]
    return col.reshape(x_pad.shape[0], -1, out_H * out_W)

def _col2im(col: cp.ndarray, x_shape: tuple, k: cp.ndarray, i: cp.ndarray, j: cp.ndarray, out_H: int, out_W: int, padding: int) -> cp.ndarray:
    batch, C_in, H, W = x_shape
    H_pad = H + 2 * padding
    W_pad = W + 2 * padding
    x_pad = cp.zeros((batch, C_in, H_pad, W_pad), dtype=col.dtype)
    col_r = col.reshape(batch, -1, out_H, out_W)
    cupyx.scatter_add(
        x_pad,
        (cp.arange(batch)[:, None, None, None], k[None, :, None, None], i[None, :, :, None], j[None, :, None, :]),
        col_r
    )
    return x_pad[:, :, padding:-padding, padding:-padding] if padding > 0 else x_pad

class Conv2D(Module):
    def __init__(self, n_filters: int, kernel_size: int, act_func: Callable, stride: int = 1, padding: int = 0, use_bn: bool = True, bn_momentum: float = 0.9, prv_layer=None, next_layer=None):
        self.n_filters = n_filters
        self.kH = self.kW = kernel_size
        self.act_func = act_func
        self.stride = stride
        self.padding = padding
        self.use_bn = use_bn
        self.bn_mom = bn_momentum
        self.prv_layer = prv_layer
        self.next_layer = next_layer
        self.network = None

    def init(self) -> None:
        C_in, H_in, W_in = self.prv_layer.out_shape
        self.out_H = (H_in + 2 * self.padding - self.kH) // self.stride + 1
        self.out_W = (W_in + 2 * self.padding - self.kW) // self.stride + 1
        self.out_shape = (self.n_filters, self.out_H, self.out_W)

        fan_in = C_in * self.kH * self.kW
        self.kernels = (cp.random.standard_normal((self.n_filters, C_in, self.kH, self.kW)) * cp.sqrt(2.0 / fan_in)).astype(cp.float32)
        self.biases = cp.zeros(self.n_filters, dtype=cp.float32)

        self._k, self._i, self._j = _im2col_indices(C_in, self.kH, self.kW, self.out_H, self.out_W, self.stride)

        if self.use_bn:
            shape = (1, self.n_filters, 1, 1)
            self.bn_gamma = cp.ones(shape, dtype=cp.float32)
            self.bn_beta = cp.zeros(shape, dtype=cp.float32)
            self.bn_run_mean = cp.zeros(shape, dtype=cp.float32)
            self.bn_run_var = cp.ones(shape, dtype=cp.float32)
            self.bn_eps = 1e-5
            self.Z_pre_bn = None
            self.x_hat = None
            self.bn_mean = None
            self.bn_var = None
            self.d_gamma = None
            self.d_beta = None

        self.col = None
        self.x_shape_cache = None
        self.Z = None
        self.layer_output = None
        self.layer_delta_term = None
        self.dW = None
        self.db = None

    def forward(self) -> None:
        x = self.prv_layer.layer_output
        batch = x.shape[0]
        self.x_shape_cache = x.shape

        x_pad = (cp.pad(x, ((0, 0), (0, 0), (self.padding, ) * 2, (self.padding, ) * 2)) if self.padding > 0 else x)
        col = _im2col(x_pad, self._k, self._i, self._j, self.out_H, self.out_W)
        self.col = col

        W_col = self.kernels.reshape(self.n_filters, -1)
        Z_col = cp.einsum('fc,bcn->bfn', W_col, col, optimize=True)
        Z_col += self.biases[None, :, None]
        Z = Z_col.reshape(batch, self.n_filters, self.out_H, self.out_W)

        if self.use_bn:
            Z = self._bn_forward(Z)
        self.Z = Z
        self.layer_output = self.act_func(Z)

    def _bn_forward(self, Z: cp.ndarray) -> cp.ndarray:
        self.Z_pre_bn = Z
        training = self.network.training if self.network else True

        if training:
            mean = Z.mean(axis=(0, 2, 3), keepdims=True)
            var = Z.var(axis=(0, 2, 3), keepdims=True)
            self.bn_mean = mean
            self.bn_var = var
            self.bn_run_mean = self.bn_mom * self.bn_run_mean + (1 - self.bn_mom) * mean
            self.bn_run_var = self.bn_mom * self.bn_run_var + (1 - self.bn_mom) * var
        else:
            mean = self.bn_run_mean
            var = self.bn_run_var

        x_hat = (Z - mean) / cp.sqrt(var + self.bn_eps)
        self.x_hat = x_hat
        return self.bn_gamma * x_hat + self.bn_beta

    def compute_delta_term(self, network, targets: cp.ndarray) -> None:
        incoming = self.next_layer.layer_delta_term
        dZ = self.act_func(self.Z, derived=True) * incoming

        if self.use_bn:
            dZ = self._bn_backward(dZ)

        batch = dZ.shape[0]
        W_col = self.kernels.reshape(self.n_filters, -1)
        dZ_col = dZ.reshape(batch, self.n_filters, -1)

        dW_col = cp.einsum('bfn,bcn->fc', dZ_col, self.col, optimize=True)
        self.dW = dW_col.reshape(self.kernels.shape)
        self.db = dZ_col.sum(axis=(0, 2))

        d_col = cp.einsum('fc,bfn->bcn', W_col, dZ_col, optimize=True)
        self.layer_delta_term = _col2im(d_col, self.x_shape_cache, self._k, self._i, self._j, self.out_H, self.out_W, self.padding)

    def _bn_backward(self, d_out: cp.ndarray) -> cp.ndarray:
        Z = self.Z_pre_bn
        mean = self.bn_mean
        var = self.bn_var
        eps = self.bn_eps
        N = d_out.shape[0] * d_out.shape[2] * d_out.shape[3]

        self.d_gamma = (d_out * self.x_hat).sum(axis=(0, 2, 3), keepdims=True)
        self.d_beta = d_out.sum(axis=(0, 2, 3), keepdims=True)

        inv_std = 1.0 / cp.sqrt(var + eps)
        dx_hat = d_out * self.bn_gamma
        diff = Z - mean

        dvar = (dx_hat * diff * (-0.5) * (var + eps)**(-1.5)).sum(axis=(0, 2, 3), keepdims=True)
        dmean = ((dx_hat * (-inv_std)).sum(axis=(0, 2, 3), keepdims=True) + dvar * (-2.0 * diff).sum(axis=(0, 2, 3), keepdims=True) / N)

        dZ = dx_hat * inv_std + dvar * 2.0 * diff / N + dmean / N
        return dZ

    def update_weights(self, lr: float) -> None:
        self.kernels -= lr * self.dW
        self.biases -= lr * self.db
        if self.use_bn:
            self.bn_gamma -= lr * self.d_gamma
            self.bn_beta -= lr * self.d_beta


# ==========================================
# 4. Pooling & Flatten
# ==========================================
class MaxPool2D(Module):
    def __init__(self, pool_size: int, stride: int, prv_layer=None, next_layer=None):
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
        x = self.prv_layer.layer_output
        batch, C, H, W = x.shape
        p, s = self.pool_size, self.stride
        self.x_shape = x.shape

        shape = (batch, C, self.out_H, self.out_W, p, p)
        st = x.strides
        strides = (st[0], st[1], s * st[2], s * st[3], st[2], st[3])
        windows = cp.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

        out = windows.max(axis=(4, 5))
        self.mask = (windows == out[:, :, :, :, None, None])
        self.layer_output = out

    def compute_delta_term(self, network, targets: cp.ndarray) -> None:
        incoming = self.next_layer.layer_delta_term
        p, s = self.pool_size, self.stride

        d_input = cp.zeros(self.x_shape, dtype=incoming.dtype)
        count = self.mask.sum(axis=(4, 5), keepdims=True).clip(min=1)
        d_windows = (incoming[:, :, :, :, None, None] * (self.mask / count))

        for oh in range(self.out_H):
            for ow in range(self.out_W):
                d_input[:, :, oh * s:oh * s + p, ow * s:ow * s + p] += d_windows[:, :, oh, ow]

        self.layer_delta_term = d_input

class Flatten(Module):
    def __init__(self, prv_layer=None, next_layer=None):
        self.prv_layer = prv_layer
        self.next_layer = next_layer
        self.layer_output = None
        self.layer_delta_term = None
        self.conv_shape = None

    def init(self) -> None:
        flat_size = int(np.prod(self.prv_layer.out_shape))
        self.out_shape = (flat_size, )
        self.n_neuron = flat_size

    def forward(self) -> None:
        x = self.prv_layer.layer_output
        self.conv_shape = x.shape
        self.layer_output = x.reshape(x.shape[0], -1)

    def compute_delta_term(self, network, targets: cp.ndarray) -> None:
        incoming = self.next_layer.layer_delta_term
        if hasattr(self.next_layer, 'weights'):
            incoming = incoming @ self.next_layer.weights
        self.layer_delta_term = incoming.reshape(self.conv_shape)


# ==========================================
# 5. Linear / Dense
# ==========================================
class Dense(Module):
    def __init__(self, n_neuron: int, act_func: Callable, use_dropout: bool = False, drop_rate: float = 0.3, weights_initialization: str = "He"):
        super().__init__()
        self.n_neuron = n_neuron
        self.act_func = act_func
        self.use_dropout = use_dropout
        self.drop_rate = drop_rate
        self.w_init_method = weights_initialization
        self.network = None

    def init(self):
        if hasattr(self.prv_layer, 'out_shape'):
            n_in = int(np.prod(self.prv_layer.out_shape))
        elif hasattr(self.prv_layer, 'n_neuron'):
            n_in = self.prv_layer.n_neuron
        else:
            n_in = self.prv_layer.layer_output.shape[-1]

        _temp = cp.random.standard_normal((self.n_neuron, n_in))
        if self.w_init_method == "He":
            self.weights = (_temp * cp.sqrt(2 / n_in)).astype(cp.float32)
        else:
            self.weights = (_temp * cp.sqrt(1 / n_in)).astype(cp.float32)

        self.biases = cp.zeros(self.n_neuron, dtype=cp.float32)
        self.out_shape = (self.n_neuron, )

    def forward(self, predict_input: cp.ndarray = None) -> cp.ndarray:
        inp = predict_input if predict_input is not None else self.prv_layer.layer_output
        self.Z = inp @ self.weights.T + self.biases
        out = self.act_func(self.Z)

        training = self.network.training if hasattr(self.network, 'training') else True
        if self.use_dropout and training:
            self.drop_mask = (cp.random.random(out.shape) > self.drop_rate).astype(cp.float32)
            out = (out * self.drop_mask) / (1.0 - self.drop_rate)

        self.layer_output = out
        return self.layer_output

    def compute_delta_term(self, network, targets: cp.ndarray) -> None:
        if self.next_layer is None:
            if getattr(self.act_func, '__name__', '') == "softmax":
                grad_loss = network.compute_loss(derived=True, targets=targets)[:, None, :]
                soft_jacobian = self.act_func(self.Z, derived=True)
                delta = (grad_loss @ soft_jacobian).reshape(grad_loss.shape[0], grad_loss.shape[-1])
            else:
                grad_loss = network.compute_loss(derived=True, targets=targets)
                delta = self.act_func(self.Z, derived=True) * grad_loss
        else:
            incoming = self.next_layer.layer_delta_term
            if hasattr(self.next_layer, 'weights'):
                incoming = incoming @ self.next_layer.weights
            elif hasattr(self.next_layer, 'conv_shape'):
                incoming = incoming

            delta = self.act_func(self.Z, derived=True) * incoming

        if self.use_dropout and (hasattr(self.network, 'training') and self.network.training):
            delta = (delta * self.drop_mask) / (1.0 - self.drop_rate)

        self.layer_delta_term = delta
        inp = self.prv_layer.layer_output
        self.dW = self.layer_delta_term.T @ inp
        self.db = cp.sum(self.layer_delta_term, axis=0)

    def update_weights(self, lr: float) -> None:
        self.weights -= lr * self.dW
        self.biases -= lr * self.db

class InputLayer(Module):
    def __init__(self, layer_input: cp.ndarray | None = None, input_shape: tuple | None = None):
        super().__init__()
        self.layer_input = layer_input
        if input_shape is None:
            self.out_shape = layer_input.shape[1:]
            self.n_neuron = layer_input.shape[1] if len(layer_input.shape) == 2 else np.prod(layer_input.shape[1:])
        else:
            self.out_shape = input_shape[1:]
            self.n_neuron = input_shape[1] if len(input_shape) == 2 else np.prod(input_shape[1:])

    def forward(self, predict_input: cp.ndarray = None) -> cp.ndarray:
        self.layer_output = predict_input if predict_input is not None else self.layer_input
        return self.layer_output


# ==========================================
# 6. Network Orchestrator
# ==========================================
class Network:
    def __init__(self, layers: list, loss_func: Callable, training_set: tuple[np.ndarray, np.ndarray] | None = None, batch: int | None = None, learning_rate: float = 0.01, lr_decay: float = 1.0, epsilon: float | None = 0.0001, epoch_limit: int | None = 10, test_set: tuple[np.ndarray, np.ndarray] | None = None, iteration_event_trigger: int | None = -1, eval_every: int = 0, exponential_moving_average: float = 0.2, augment_fn: Callable | None = None):
        if training_set:
            self.x_train = cp.array(training_set[0])
            self.y_train = cp.array(training_set[1])
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
            self.x_test = cp.array(test_set[0])
            self.y_test = cp.array(test_set[1])

        self.network_loss = 0
        self.iterations = 0
        self.epoch = 0
        self.training = True

        self._init_network()

    def _init_network(self) -> None:
        indexes = len(self.network_layers)
        for i in range(indexes):
            if i > 0:
                self.network_layers[i].prv_layer = self.network_layers[i - 1]
            if i < indexes - 1:
                self.network_layers[i].next_layer = self.network_layers[i + 1]
            self.network_layers[i].network = self
            self.network_layers[i].init()
        print("--- Neural network is initialized successfully ---")

    def compute_loss(self, targets: cp.ndarray, derived: bool = False) -> cp.ndarray:
        predicted_vals = self.network_layers[-1].layer_output
        loss = self.loss_func(predicted_values=predicted_vals, targets=targets, derived=derived)
        if not derived:
            self.network_loss = loss
        return loss

    def _forward_propagation(self, predict_input: cp.ndarray = None) -> None:
        self.network_layers[0].forward(predict_input)
        for layer in self.network_layers[1:]:
            layer.forward()

    def _backward_propagation(self, targets: cp.ndarray, current_lr: float = None) -> None:
        if current_lr is None:
            current_lr = self.learning_rate
        for layer in self.network_layers[-1:0:-1]:
            layer.compute_delta_term(network=self, targets=targets)
        for layer in self.network_layers[1:]:
            layer.update_weights(current_lr)

    def fit_model(self) -> None:
        if self.eval_every == 0 and not self.test_set_exist:
            pass
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
            indices = cp.random.permutation(n_samples)
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
                    smoothed_loss_diff = abs(prv_smoothed_loss - smoothed_loss) / smoothed_loss
                    if smoothed_loss_diff < self.epsilon:
                        stop = True
                        break

                if self.iet > 0 and self.iterations % self.iet == 0:
                    print(f"Updates: #{self.iterations} | Loss: {float(new_loss):.6f} | Epoch: #{self.epoch} | clr = {current_lr:.4f}")
                self.iterations += 1
                prv_smoothed_loss = smoothed_loss

            if self.test_set_exist and self.eval_every > 0 and self.epoch % self.eval_every == 0:
                acc = self.evaluate()
                self.training = True
                if acc is not None:
                    print(f"Epoch #{self.epoch} | Test evaluation: {acc:.2f}%")

    def predict(self, input: np.ndarray) -> np.ndarray:
        self.training = False
        batch_size = self.batch if hasattr(self, 'batch') and self.batch is not None else 256
        results = []
        n_samples = input.shape[0]
        for i in range(0, n_samples, batch_size):
            batch_input = cp.array(input[i:i + batch_size])
            self._forward_propagation(predict_input=batch_input)
            results.append(cp.asnumpy(self.network_layers[-1].layer_output))
        return np.concatenate(results, axis=0)

    def evaluate(self) -> float:
        self.training = False
        model = getattr(self.loss_func, '__name__', 'unknown')
        if model == "cc_loss":
            act_classes = np.argmax(cp.asnumpy(self.y_test), axis=1)
            predictions = self.predict(input=cp.asnumpy(self.x_test))
            pred_classes = np.argmax(predictions, axis=1)
            precision = np.mean(pred_classes == act_classes) * 100
            return precision
        elif model == "MSE":
            if not self.test_set_exist:
                print("--- Warming, model does not have test set for evaluation ---")
                return None
            else:
                y_test_cpu = cp.asnumpy(self.y_test)
                act_mean = np.mean(y_test_cpu)
                sst = np.sum((y_test_cpu - act_mean)**2)
                predictions = self.predict(input=cp.asnumpy(self.x_test))
                ssr = np.sum((y_test_cpu - predictions)**2)
                return (1 - ssr / sst) * 100
        return None

    def save_weights(self, path: str, accuracy: float = None) -> None:
        arrays = {}
        for idx, layer in enumerate(self.network_layers[1:], 1):
            if hasattr(layer, 'weights') and layer.weights is not None:
                arrays[f"l{idx}_weights"] = cp.asnumpy(layer.weights)
                arrays[f"l{idx}_biases"] = cp.asnumpy(layer.biases)
            if hasattr(layer, 'kernels') and layer.kernels is not None:
                arrays[f"l{idx}_kernels"] = cp.asnumpy(layer.kernels)
                arrays[f"l{idx}_biases"] = cp.asnumpy(layer.biases)
            if hasattr(layer, 'use_bn') and layer.use_bn:
                arrays[f"l{idx}_bn_gamma"] = cp.asnumpy(layer.bn_gamma)
                arrays[f"l{idx}_bn_beta"] = cp.asnumpy(layer.bn_beta)
                arrays[f"l{idx}_bn_run_mean"] = cp.asnumpy(layer.bn_run_mean)
                arrays[f"l{idx}_bn_run_var"] = cp.asnumpy(layer.bn_run_var)
        if accuracy is not None:
            arrays["accuracy"] = np.float32(accuracy)
        np.savez(path, **arrays)
        acc_str = f"  |  accuracy: {accuracy:.2f}%" if accuracy is not None else ""
        print(f"Weights saved -> {path}.npz{acc_str}")

    def load_weights(self, path: str) -> float | None:
        data = np.load(path if path.endswith('.npz') else path + '.npz')
        start_idx = 0 if any(k.startswith('l0_') for k in data.files) else 1

        for idx, layer in enumerate(self.network_layers[1:], start_idx):
            if f"l{idx}_weights" in data:
                layer.weights = cp.array(data[f"l{idx}_weights"])
                layer.biases = cp.array(data[f"l{idx}_biases"])
            if f"l{idx}_kernels" in data:
                layer.kernels = cp.array(data[f"l{idx}_kernels"])
                layer.biases = cp.array(data[f"l{idx}_biases"])
            if f"l{idx}_bn_gamma" in data:
                layer.bn_gamma = cp.array(data[f"l{idx}_bn_gamma"])
                layer.bn_beta = cp.array(data[f"l{idx}_bn_beta"])
                layer.bn_run_mean = cp.array(data[f"l{idx}_bn_run_mean"])
                layer.bn_run_var = cp.array(data[f"l{idx}_bn_run_var"])

        accuracy = float(data["accuracy"]) if "accuracy" in data else None
        acc_str = f"  |  accuracy: {accuracy:.2f}%" if accuracy is not None else ""
        print(f"Weights loaded <- {path}{acc_str}")
        return accuracy
