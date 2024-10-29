from types import SimpleNamespace
from beras.activations import ReLU, LeakyReLU, Softmax
from beras.layers import Dense
from beras.losses import CategoricalCrossEntropy, MeanSquaredError
from beras.metrics import CategoricalAccuracy
from beras.onehot import OneHotEncoder
from beras.optimizers import Adam
from preprocess import load_and_preprocess_data
import numpy as np

from beras.model import SequentialModel

if __name__ == '__main__':
  path = "predictions_.npy"
  # TODO, Train a model!
  X_train, y_train, X_test, y_test = load_and_preprocess_data()
  print(X_train.shape)
  # X_train = np.random.shuffle(X_train)
  # rng = np.random.default_rng()
  # rng.shuffle(X_train)
  print(X_train.shape)
  one_hot = OneHotEncoder()
  y_train = one_hot(y_train)
  y_test = one_hot(y_test)
  layer_output_size = 128
  # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
  model = SequentialModel(
    [
      Dense(input_size = 28*28, output_size = layer_output_size, initializer="xavier"),
      ReLU(),
      Dense(input_size = layer_output_size, output_size = layer_output_size, initializer="xavier"),
      ReLU(),
      Dense(input_size = layer_output_size, output_size = 10, initializer="xavier"),
      Softmax()
    ]
  )
  model.compile(
    optimizer = Adam(0.001), loss_fn = CategoricalCrossEntropy(), acc_fn = CategoricalAccuracy()
  )
  model.fit(X_train, y_train, epochs=10, batch_size=128)
  pred = model.evaluate(X_test, y_test, batch_size=128)
  pred = one_hot.inverse(pred)
  np.save(path, pred)