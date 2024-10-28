import numpy as np
from beras.layers import *
from beras.losses import MeanSquaredError, CategoricalCrossEntropy
from beras.activations import *
from tensorflow.keras.losses import categorical_crossentropy

y_true = [[[0, 1, 0], [0, 0, 1]], [[0, 1, 0], [0, 0, 1]]]
y_pred = [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]], [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]]
loss = categorical_crossentropy(y_true, y_pred)
# assert loss.shape == (2,)
print(loss)


# yt = np.array([[1,0,0],[0,1,0],[0,0,1]])
# yp = np.array([[1,0,0], [0,1,0], [0,0,1]])
# y_true=np.array([[[3],[3],[4]], [[1],[2],[3]]])
# y_pred=np.array([[[1],[2],[3]], [[3],[2],[1]]])

# s = np.square(y_pred-y_true)
# m = (np.mean(s, keepdims=True))
# print(m)
# print(m.shape)

# m = CategoricalCrossEntropy()
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



