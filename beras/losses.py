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
        obs_mean = np.mean(np.square(y_pred - y_true), axis = -1)
        batch_mean = np.mean(obs_mean, keepdims=True)
        return Tensor(batch_mean)
    def get_input_gradients(self) -> list[Tensor]:
        y_pred = self.inputs[0]
        y_true = self.inputs[1]
        return [Tensor(2 * (y_pred - y_true) / np.prod(y_pred.shape)), Tensor(np.zeros_like(y_true))]

class CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""
        # return NotImplementedError
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        y_true = y_true.reshape(-1, y_true.shape[-1])
        # loss = np.mean(-np.sum(y_true * np.log(y_pred), axis=-1), keepdims=True)
        loss = np.sum(-np.sum(y_true * np.log(y_pred), axis=-1, keepdims=True), keepdims=True) / y_pred.shape[0]
        return Tensor(loss)
        
    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        # return NotImplementedError
        y_pred = self.inputs[0]
        y_true = self.inputs[1]
        grad = -(y_true / y_pred) 
        # grad = (1 / self.y_pred.shape[0]) * -(self.y_true * self.y_pred)
        # grad_y_true =  np.zeros_like(self.y_true)
        return [Tensor(grad), Tensor(np.zeros_like(grad))]
     # (128, 10) (128, 10)


