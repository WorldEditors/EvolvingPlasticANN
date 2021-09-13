"""
Assembly of Layers to Give A plastic Neural Networks
"""
import numpy
from layers import Layers

class PlasticModels(object):
    def __init__(self):
        self._parameters = dict()
        self._evolution_noises = dict()
        self._layers = dict()

    def initialize_parameters(self):
        for key in self._layers:
            self._layers[key].initialize_layer(self)

    def add_layer(self, key, layer):
        self._layers[key] = layer

    def get_layer(self, key):
        return self._layers[key]

    def get_parameter(self, key):
        return self._parameters[key]

    def mod_parameter(self, key, mod):
        self._parameters[key] += mod

    def get_layer_hidden(self, key):
        return self._layers[key].hidden

    def add_parameter(self, key, param, noise):
        self._parameters[key] = numpy.copy(param)
        self._evolution_noises[key] = noise

    def has_parameter(self, key):
        return (key in self._parameters)

    def reset(self):
        for key in self._layers:
            sel._layers.reset()

    def forward(self, obs, heb_is_on):
        if("observation" not in self._layers or "output" not in self._layers):
            raise Exception("A model must have observation layer and output layer")
        self._layers["observation"].set(obs)
        for key in self._layers:
            self._layers[key](self, heb_is_on)

        return self._layers["output"].hidden 

    def forward_batch(self, obs):
        if("observation" not in self._layers or "output" not in self._layers):
            raise Exception("A model must have observation layer and output layer")
        self._layers["observation"].set(obs)
        for key in self._layers:
            self._layers[key].forward_batch(self)

        return self._layers["output"].hidden 

    def backward_batch(self, grads):
        if("observation" not in self._layers or "output" not in self._layers):
            raise Exception("A model must have observation layer and output layer")
        self._layers["output"].backward_batch(grads, self)
