from collections import defaultdict
import numpy as np
from beras.core import Diffable, Tensor

class GradientTape:
    def __init__(self):
        # Dictionary mapping the object id of an output Tensor to the Diffable layer it was produced from.
        self.previous_layers: defaultdict[int, Diffable | None] = defaultdict(lambda: None)
    def __enter__(self):
        # When tape scope is entered, all Diffables will point to this tape.
        if Diffable.gradient_tape is not None:
            raise RuntimeError("Cannot nest gradient tape scopes.")
        Diffable.gradient_tape = self
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        # When tape scope is exited, all Diffables will no longer point to this tape.
        Diffable.gradient_tape = None
    def gradient(self, target: Tensor, sources: list[Tensor]) -> list[Tensor]:
        """
        Computes the gradient of the target tensor with respect to the sources.
        :param target: the tensor to compute the gradient of, typically loss output
        :param sources: the list of tensors to compute the gradient with respect to
        In order to use tensors as keys to the dictionary, use the python built-in ID function here: https://docs.python.org/3/library/functions.html#id.
        To find what methods are available on certain objects, reference the cheat sheet
        """
        ### TODO: Populate the grads dictionary with {weight_id, weight_gradient} pairs.
        queue = [target]                    ## Live queue; will be used to propagate backwards via breadth-first-search.
        grads = defaultdict(lambda: None)   ## Grads to be recorded. Initialize to None. Note: stores {id: list[gradients]}
        # Use id(tensor) to get the object id of a tensor object.
        # in the end, your grads dictionary should have the following structure:
        # {id(tensor): [gradient]}
        # What tensor and what gradient is for you to implement!
        # compose_input_gradients and compose_weight_gradients are methods that will be helpful
        grads[id(target)] = [np.ones_like(target)]
        gradient_list = []
        while queue:
            curr = queue.pop(0)
            prev = self.previous_layers[id(curr)]
            # print(prev.__class__)
            if prev is None:
                continue
            weight_grad = prev.compose_weight_gradients(grads[id(curr)])
            # print("grad:", grads[id(curr)])
            input_grad = prev.compose_input_gradients(grads[id(curr)])
            print("weight grad:", weight_grad)
            print("input grad:", input_grad)
            for weight, grad in zip(prev.trainable_variables, weight_grad):
                grads[(id(weight))] = [grad] if grads[id(weight)] is None else grads[(id(weight))] + [grad]
            for inputs, grad in zip(prev.inputs, input_grad):
                if id(inputs) not in queue:
                    queue.append(id(inputs))
                grads[(id(inputs))] = [grad] if grads[id(inputs)] is None else grads[(id(inputs))] + [grad]
            for source in sources:
                if grads[id(source)] is not None:
                    gradient_list.append(Tensor.sum(grads[id(source)], axis = 0))
                else:
                    gradient_list.append(None)
            return gradient_list
