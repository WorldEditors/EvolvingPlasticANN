"""
Assembly of Layers to Give A plastic Neural Networks
"""
import numpy
import pickle
from epann.layers import *
from epann.parameters import Parameters

class Models(Parameters):
    def __init__(self, **kwargs):
        super(Models, self).__init__()
        for key in kwargs:
            self.__dict__[key] = kwargs[key]
        self._layers = list()
        self.build_model()

    def build_model(self):
        raise NotImplementedError("Must specify build model")

    def add_layer(self, layer_type, **kw_args):
        self._layers.append(layer_type(params=self, **kw_args))
        return self._layers[-1]

    def clear_layers(self):
        self._layers = list()
        self._parameters = dict()

    # reset the model to initial_states
    def reset(self):
        for layer in self._layers:
            layer.reset()

    def forward(self, inputs):
        raise NotImplementedError("forward function not implemented")

    def __call__(self, inputs):
        return self.forward(inputs)

    def load(self, file_name):
        file_op = open(file_name, "rb")
        self.set_all_parameters(pickle.load(file_op))
        file_op.close()
        self.reset()

    def save(self, file_name):
        file_op = open(file_name, "wb")
        pickle.dump(self._parameters, file_op)
        file_op.close()

    def __repr__(self):
        return self.parameters()

    def backward_batch(self, grads):
        raise NotImplementedError("backward function not implemented")

    def forward_batch(self, grads):
        raise NotImplementedError("forward function not implemented")

    # Policy Gradient for Continuous action space
    def backward_policy_gradient_continuous(self, alpha, inputs, actions, advantage):
        outputs = self.forward_batch(inputs)
        grads = alpha * numpy.expand_dims(advantage, axis=1) * (outputs - actions)
        self.backward_batch(grads)

    # Back propagation (Policy Gradient for discrete action space)
    def backward_policy_gradient_discrete(self, alpha, inputs, actions, advantage):
        outputs = self.forward_batch(inputs)
        idxes = numpy.vstack([numpy.arange(actions.shape[0]), actions])
        sigma = numpy.zeros_like(outputs)
        sigma[idxes.T] = 1.0
        grads = alpha * numpy.expand_dims(advantage, axis=1) * (outputs - sigma)
        self.backward_batch(grads)

    # Back propagation (Supservised Learning)
    def backward_supervised_learning(self, alpha, inputs, labels):
        outputs = self.forward_batch(inputs)
        grads = 2.0 * alpha * numpy.clip(outputs - labels, -1.0, 1.0)
        self.backward_batch(grads)
