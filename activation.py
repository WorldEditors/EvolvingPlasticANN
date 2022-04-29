"""
Dicts for Activation Function
"""
import sys
import numpy

def softmax(x):
    w = numpy.exp(x)
    return w/numpy.sum(w)

def grad_softmax(x):
    return x * (1.0 - x)

def tanh(x):
    return numpy.tanh(x)

def grad_tanh(x):
    return 1.0 - x * x

def sigmoid(x):
    return 0.5 * (numpy.tanh(0.5 * x) + 1.0)

def grad_sigmoid(x):
    return x * (1.0 - x)

def relu(x):
    return numpy.maximum(x, 0)

def grad_relu(x):
    return (x > 0).astype("float32")

def softsign(x):
    return x * numpy.reciprocal(numpy.abs(x) + 1)

def grad_softsign(x):
    ax = numpy.abs(x)
    return ax * numpy.reciprocal(ax + 1)

def step(x):
    return asarray(x>0, dtype="float32")

def none(x):
    return x

def grad_none(x):
    return 1.0

"""
Add New Activation Functions Here
"""

FUNC_LIST = [method for method in dir() if method.startswith('__') is False]
CUR_MODULE = sys.modules[__name__]

class ActFunc(object):
    def __init__(self, act_type):
        self._act_type = act_type
        if(act_type not in FUNC_LIST):
            raise Exception("No such activation function type: %s"%act_type)
        self._func = getattr(CUR_MODULE, act_type)
        if("grad_" + act_type not in FUNC_LIST):
            self._grad_func==None
        else:
            self._grad_func = getattr(CUR_MODULE, "grad_" + act_type)

    def __call__(self, x):
        return (self._func)(x)

    def grad(self, x):
        if(self._grad_func is None):
            raise Exception("Gradient of current activation is not supported: %s" % self._act_type)
        return self._grad_func(x)
