import numpy as np
from beras.layers import *
from beras.losses import MeanSquaredError, CategoricalCrossEntropy

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
x = np.random.normal(size=(10,10))
d = Dense(10, 5)
y = d(x)
print(y.shape)
print(d.get_input_gradients()[0].shape)
print(d.get_weight_gradients()[0].shape)
print(d.get_weight_gradients()[1].shape)
print(d.inputs.__len__())



