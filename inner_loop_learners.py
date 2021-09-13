"""
Inner Loop Learners
"""
import sys
import numpy
import pickle
import json
from numpy import random
from copy import copy, deepcopy
from models import PlasticModels
from layers import Layers

class InnerLoopLearner(object):
    def __init__(self, model_config):
        self._model = PlasticModels()
        for key in model_config:
            if(isinstance(model_config[key], Layers)):
                self._model.add_layer(key, model_config[key])
            else:
                self._model.add_parameter(key, model_config[key]["initial_parameter"], model_config[key]["noise"])
        self._model.initialize_parameters()

    def inner_loop_reset(self):
        self._model.reset()

    def get(self):
        return deepcopy(model._parameters)

    def set_params(self, params):
        self._model._parameters = deepcopy(params)

    def load(self, file_name):
        file_op = open(file_name, "rb")
        self._model._parameters = pickle.load(file_op)
        file_op.close()
        self.inner_loop_reset()

    def save(self, file_name):
        file_op = open(file_name, "wb")
        pickle.dump(self._model._parameters, file_op)
        file_op.close()

    def __repr__(self):
        return self.get()

    def run_forward(self, obs, heb_is_on=False):
        return self._model.forward(obs, heb_is_on)

    # Policy Gradient for Continuous action space
    def run_policy_gradient_continuous(self, alpha, inputs, actions, advantage):
        outputs = self._model.forward_batch(inputs)
        grads = alpha * numpy.expand_dims(advantage, axis=1) * (outputs - actions)
        self._model.backward_batch(grads)

    # Back propagation (Policy Gradient for discrete action space)
    def run_policy_gradient_discrete(self, alpha, inputs, actions, advantage):
        outputs = self._model.forward_batch(inputs)
        idxes = numpy.vstack([numpy.arange(actions.shape[0]), actions])
        sigma = numpy.zeros_like(outputs)
        sigma[idxes.T] = 1.0
        grads = alpha * numpy.expand_dims(advantage, axis=1) * (outputs - sigma)
        self._model.backward_batch(grads)

    # Back propagation (Supservised Learning)
    def run_supervised_learning(self, alpha, inputs, labels):
        outputs = self._model.forward_batch(inputs)
        grads = 2.0 * alpha * numpy.clip(outputs - labels, -1.0, 1.0)
        self._model.backward_batch(grads)
