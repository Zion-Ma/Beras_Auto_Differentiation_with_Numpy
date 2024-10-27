import numpy as np
from beras.layers import *
from beras.losses import MeanSquaredError, CategoricalCrossEntropy
from beras.activations import *

# yt = np.array([[1,0,0],[0,1,0],[0,0,1]])
# yp = np.array([[1,0,0], [0,1,0], [0,0,1]])
# # yt=np.array([[2],[3],[4]])
# # yp=np.array([[1],[2],[3]])

# m = CategoricalCrossEntropy()
# # m = MeanSquaredError()
# loss = m(yp, yt)
# print(loss)
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

x = np.array([[1,1,2],[1,1,2]])
m = np.max(x, axis = -1, keepdims=True)
x = x - m
# print(np.sum(x, axis=-1, keepdims=True))
softmax_output = x / np.sum(x, axis=-1, keepdims=True)
print(softmax_output)
print("_____________________")
for i in range(softmax_output.shape[0]):
    j = np.diagflat(softmax_output[i]) - np.outer(softmax_output[i], softmax_output[i].T)
    print(j)
# print(softmax_output)
# print(sx)
"""
[]
"""



