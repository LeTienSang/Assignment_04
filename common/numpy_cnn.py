from __future__ import annotations

import numpy as np


class NumpyConv1D:
    """Minimal educational Conv1D + ReLU + max-pooling primitives."""

    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0, np.sqrt(2 / (channels_in * kernel_size)), (channels_out, channels_in, kernel_size))
        self.bias = np.zeros(channels_out)
        self.cache = None

    def forward(self, values: np.ndarray) -> np.ndarray:
        batch, channels, length = values.shape
        kernel_size = self.weights.shape[-1]
        padded = np.pad(values, ((0, 0), (0, 0), (1, 1)), mode="constant")
        output = np.zeros((batch, self.weights.shape[0], length))
        for index in range(length):
            window = padded[:, :, index:index + kernel_size]
            output[:, :, index] = np.einsum("bck,ock->bo", window, self.weights) + self.bias
        self.cache = (values, padded)
        return output

    def backward(self, gradient: np.ndarray, learning_rate: float = 1e-3) -> np.ndarray:
        values, padded = self.cache
        batch, channels, length = values.shape
        kernel_size = self.weights.shape[-1]
        weight_gradient = np.zeros_like(self.weights)
        bias_gradient = gradient.sum(axis=(0, 2))
        input_gradient = np.zeros_like(padded)
        for index in range(length):
            window = padded[:, :, index:index + kernel_size]
            weight_gradient += np.einsum("bo,bck->ock", gradient[:, :, index], window)
            input_gradient[:, :, index:index + kernel_size] += np.einsum("bo,ock->bck", gradient[:, :, index], self.weights)
        self.weights -= learning_rate * weight_gradient / batch
        self.bias -= learning_rate * bias_gradient / batch
        return input_gradient[:, :, 1:-1]


def relu(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = values > 0
    return np.maximum(values, 0), mask


def max_pool1d(values: np.ndarray, size: int = 2) -> tuple[np.ndarray, np.ndarray]:
    trimmed = values[:, :, : values.shape[-1] // size * size]
    reshaped = trimmed.reshape(values.shape[0], values.shape[1], -1, size)
    indices = reshaped.argmax(axis=-1)
    return reshaped.max(axis=-1), indices