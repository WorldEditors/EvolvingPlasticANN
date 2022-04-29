"""
Managing Parameters
"""
import numpy
import sys
from copy import deepcopy

class Parameters(object):
    def __init__(self):
        self._parameters = dict()
        self._static_parameters = dict()

    def get(self, key, is_static=False):
        if(not is_static):
            return self._parameters[key]
        else:
            return self._static_parameters[key]

    def set(self, key, value, is_static=False):
        if(not is_static):
            self._parameters[key] = numpy.copy(value)
        else:
            self._static_parameters[key] = numpy.copy(value)

    def mod(self, key, mod, is_static=False):
        if(not is_static):
            self._parameters[key] += mod
        else:
            self._static_parameters[key] += mod

    def delete(self, key, is_static=False):
        if(not is_static):
            del self._parameters[key]
        else:
            del self._static_parameters[key]

    def add(self, key, param, is_static=False):
        if(not is_static):
            if(key not in self._parameters):
                self._parameters[key] = numpy.copy(param)
            else:
                if(param.shape == self._parameters[key].shape):
                    sys.stdout.write("WARNING: parameters are shared under key %s\n" % key)
                else:
                    raise Exception("Shared parameters must have the same shape, received %s new shape %s != %s"%(
                        key, param.shape, self._parameters[key].shape
                        ))
        else:
            if(key not in self._static_parameters):
                self._static_parameters[key] = numpy.copy(param)
            else:
                if(param.shape == self._static_parameters[key].shape):
                    sys.stdout.write("WARNING: parameters are shared under key %s\n" % key)
                else:
                    raise Exception("Shared parameters must have the same shape, received %s new shape %s != %s"%(
                        key, param.shape, self._static_parameters[key].shape
                        ))


    def set_all_parameters(self, parameter_list, is_static=False):
        if(not is_static):
            self._parameters = deepcopy(parameter_list)
        else:
            self._static_parameters = deepcopy(parameter_list)

    @property
    def parameters(self):
        return self._parameters

    @property
    def static_parameters(self):
        return self._static_parameters

    @property
    def to_vector(self):
        vector = []
        parameter_shapes = []
        for key in sorted(list(self._parameters.keys())):
            parameter_shapes.append((key, self._parameters[key].shape))
            vector.append(numpy.reshape(self._parameters[key], (numpy.product(self._parameters[key].shape),)))
        return numpy.concatenate(vector, axis=0), parameter_shapes

    @property
    def to_vector_static(self):
        vector = []
        parameter_shapes = []
        for key in sorted(list(self._static_parameters.keys())):
            parameter_shapes.append((key, self._static_parameters[key].shape))
            vector.append(numpy.reshape(self._static_parameters[key], (numpy.product(self._static_parameters[key].shape),)))
        return numpy.concatenate(vector, axis=0), parameter_shapes

    @property
    def para_segments(self):
        vector = []
        for idx, key in enumerate(sorted(list(self._parameters.keys()))):
            vector.append(numpy.full(fill_value = idx, shape=(numpy.product(self._parameters[key].shape), ), dtype="int32"))
        return numpy.concatenate(vector, axis=0)

    def from_vector(self, vector, shapes):
        self._parameters = dict()
        start = 0
        for key, shape in shapes:
            end = start + numpy.product(shape)
            self._parameters[key] = numpy.reshape(vector[start:end], shape)
            start = end
        if(end != len(vector)):
            raise Exception("Mismatch shape info and vector. Len of shape: %d; Len of vector: %d"%(end, len(vector)))

    def from_vector_static(self, vector, shapes):
        self._static_parameters = dict()
        start = 0
        for key, shape in shapes:
            end = start + numpy.product(shape)
            self._static_parameters[key] = numpy.reshape(vector[start:end], shape)
            start = end
        if(end != len(vector)):
            raise Exception("Mismatch shape info and vector. Len of shape: %d; Len of vector: %d"%(end, len(vector)))
