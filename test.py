import numpy as np
from beras.layers import *
from beras.losses import MeanSquaredError, CategoricalCrossEntropy
from beras.activations import *


# y_true = np.array([[1,0,0],[0,1,0],[0,0,1]])
# y_pred = np.array([[0,1,0], [0,1,0], [0,0,1]])
# y_true=np.array([[[3],[3],[4]], [[1],[2],[3]]])
# y_pred=np.array([[[1],[2],[3]], [[3],[2],[1]]])

# s = np.square(y_pred-y_true)
# m = (np.mean(s, keepdims=True))
# print(m)
# print(m.shape)

m = CategoricalCrossEntropy()
# m = MeanSquaredError()
# loss = m(y_pred, y_true)
# print(loss)
# print(loss.shape)
# print(loss.shape)
# print(m.get_input_gradients())
# x = np.random.normal(size=(10,10))
# d = Dense(10, 5)
# y = d(x)
# print(y.shape)
# print(d.inputs)
# print(d.get_input_gradients()[0].shape)
# print(d.get_weight_gradients()[0].shape)
# print(d.get_weight_gradients()[1].shape)
"""
[]
"""

x = np.random.normal(size=(128,10))
y = np.random.normal(size=(128,5))
print(x + y)



