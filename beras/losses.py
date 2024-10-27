import numpy as np

from beras.core import Diffable, Tensor

# import tensorflow as tf


class Loss(Diffable):
    @property
    def weights(self) -> list[Tensor]:
        return []

    def get_weight_gradients(self) -> list[Tensor]:
        return []


class MeanSquaredError(Loss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # return NotImplementedError
        self.y_pred = y_pred
        self.y_true = y_true
        mse = np.mean(np.square(y_pred - y_true), axis=-1, keepdims=True)
        return Tensor(mse)

    def get_input_gradients(self) -> list[Tensor]:
        # return NotImplementedError
        grad = (2 / self.y_pred.shape[0]) * (self.y_pred - self.y_true)
        # grad_y_true =  np.zeros_like(self.y_true)
        return [Tensor(grad)]


class CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""
        # return NotImplementedError
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        self.y_true = y_true
        self.y_pred = y_pred
        loss = np.mean(-np.sum(y_true * np.log(y_pred), axis=-1, keepdims=True), keepdims=True)
        return Tensor(loss)
        
    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        # return NotImplementedError
        grad = (1 / self.y_pred.shape[0]) * -(self.y_true * self.y_pred)
        # grad_y_true =  np.zeros_like(self.y_true)
        return [Tensor(grad)]


