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
        assert(y_pred.shape == y_true.shape)
        y_pred = np.atleast_2d(y_pred)
        y_true = np.atleast_2d(y_true)
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        y_true = y_true.reshape(-1, y_true.shape[-1])
        self.y_pred = y_pred
        self.y_true = y_true
        return 0.5*np.sum((y_pred-y_true)**2)/y_pred.shape[0]
    def get_input_gradients(self) -> list[Tensor]:
        return [(self.y_pred-self.y_true)/np.prod(self.y_pred.shape[:-1])]
    # def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # # return NotImplementedError
        # self.y_pred = y_pred
        # self.y_true = y_true
        #         # Compute the squared difference
        # squared_diff = (y_pred - y_true) ** 2

        # # Take the mean across each example (axis=-1)
        # example_means = np.mean(squared_diff, axis=-1, keepdims=True)

        # # # Then take the mean across the batch
        # batch_mean = np.mean(example_means, axis = -1, keepdims=True) / 2
        
        # return Tensor(batch_mean)
        # # mse = np.mean(np.mean(np.square(y_pred - y_true), axis=-1), keepdims=True)
        # mse = np.mean(np.square(y_pred-y_true), axis=0)
        # return Tensor(mse)

    # def get_input_gradients(self) -> list[Tensor]:
    #     # return NotImplementedError
    #     grad = (2 / self.y_pred.shape[0]) * (self.y_pred - self.y_true)
    #     # grad_y_true =  np.zeros_like(self.y_true)
    #     return [Tensor(grad)]


class CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""
        # return NotImplementedError
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        y_true = y_true.reshape(-1, y_true.shape[-1])
        self.y_true = y_true
        self.y_pred = y_pred
        # loss = np.mean(-np.sum(y_true * np.log(y_pred), axis=-1), keepdims=True)
        loss = np.sum(-np.sum(y_true * np.log(y_pred), axis=-1)) / y_pred.shape[0]
        return Tensor(loss)
        
    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        # return NotImplementedError
        grad = (1 / self.y_pred.shape[0]) * -(self.y_true * self.y_pred)
        # grad_y_true =  np.zeros_like(self.y_true)
        return [Tensor(grad)]


