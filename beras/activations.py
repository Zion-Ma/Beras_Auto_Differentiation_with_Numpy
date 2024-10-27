import numpy as np

from .core import Diffable,Tensor

class Activation(Diffable):
    @property
    def weights(self): return []

    def get_weight_gradients(self): return []


################################################################################
## Intermediate Activations To Put Between Layers

class LeakyReLU(Activation):

    ## TODO: Implement for default intermediate activation.

    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def forward(self, x) -> Tensor:
        """Leaky ReLu forward propagation!"""
        self.x = x
        return Tensor(np.where(x >= 0, x, self.alpha * x))
        

    def get_input_gradients(self) -> list[Tensor]:
        """
        Leaky ReLu backpropagation!
        To see what methods/variables you have access to, refer to the cheat sheet.
        Hint: Make sure not to mutate any instance variables. Return a new list[tensor(s)]
        """
        # raise NotImplementedError
        return [Tensor(np.where(self.x >= 0, 1, self.alpha))]

    def compose_input_gradients(self, J):
        return self.get_input_gradients()[0] * J

class ReLU(LeakyReLU):
    ## GIVEN: Just shows that relu is a degenerate case of the LeakyReLU
    def __init__(self):
        super().__init__(alpha=0)


################################################################################
## Output Activations For Probability-Space Outputs

class Sigmoid(Activation):
    
    ## TODO: Implement for default output activation to bind output to 0-1
    
    def forward(self, x) -> Tensor:
        # raise NotImplementedError
        self.sig_output = 1 / (1 + np.exp(-x))
        return Tensor(1 / (1 + np.exp(-x)))

    def get_input_gradients(self) -> list[Tensor]:
        """
        To see what methods/variables you have access to, refer to the cheat sheet.
        Hint: Make sure not to mutate any instance variables. Return a new list[tensor(s)]
        """
        # raise NotImplementedError
        return [self.sig_output * (1 - self.sig_output)]
    def compose_input_gradients(self, J):
        return self.get_input_gradients()[0] * J


class Softmax(Activation):
    # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/

    ## TODO [2470]: Implement for default output activation to bind output to 0-1

    def forward(self, x):
        """Softmax forward propagation!"""
        ## Not stable version
        ## exps = np.exp(inputs)
        ## outs = exps / np.sum(exps, axis=-1, keepdims=True)

        ## HINT: Use stable softmax, which subtracts maximum from
        ## all entries to prevent overflow/underflow issues
        # raise NotImplementedError
        logit = x - np.max(x, axis=-1, keepdims=True)
        exp_logit = np.exp(logit)
        softmax_output = exp_logit / np.sum(exp_logit, axis=-1, keepdims=True)
        return Tensor(softmax_output)

    def get_input_gradients(self):
        """Softmax input gradients!"""
        # https://stackoverflow.com/questions/48633288/how-to-assign-elements-into-the-diagonal-of-a-3d-matrix-efficiently
        x, y = self.inputs + self.outputs
        bn, n = x.shape
        grad = np.zeros(shape=(bn, n, n), dtype=x.dtype)
        # TODO: Implement softmax gradient
        # raise NotImplementedError
        for i in range(bn):
            logit = x[i] - np.max(x[i])
            exp_logit = np.exp(logit)
            softmax_output = exp_logit / np.sum(exp_logit, axis=-1, keepdims=True)
            jacobian = np.diagflat(softmax_output) - np.dot(softmax_output, softmax_output.T)
            grad[i,:,:] = jacobian
        return [Tensor(grad)]
    