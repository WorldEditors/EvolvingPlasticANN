#Plasticity Rules
import numpy

def Plasticity_ABCD(inputs, outputs, A, B, C, D):
    x = numpy.expand_dims(inputs, axis=0)
    y = numpy.expand_dims(outputs, axis=1) 
    _x = numpy.ones_like(x)
    _y = numpy.ones_like(y)

    return (A * numpy.matmul(y, x) + B * numpy.matmul(_y, x) 
            + C * numpy.matmul(y, _x) + D * numpy.matmul(_y, _x)
            )

def Plasticity_SABCD(inputs, outputs, S, A, B, C, D):
    x = numpy.expand_dims(inputs, axis=0)
    y = numpy.expand_dims(outputs * S, axis=1) 
    _x = numpy.ones_like(x)
    _y = numpy.expand_dims(S, axis=1)

    return (A * numpy.matmul(y, x) + B * numpy.matmul(_y, x) 
            + C * numpy.matmul(y, _x) + D * numpy.matmul(_y, _x)
            )

def Plasticity_S(inputs, outputs, S, A, B, C, D):
    x = numpy.expand_dims(inputs, axis=0)
    y = numpy.expand_dims(outputs * S, axis=1) 
    _x = numpy.ones_like(x)
    _y = numpy.expand_dims(S, axis=1)

    return (A * numpy.matmul(y, x) + B * numpy.matmul(_y, x) 
            + C * numpy.matmul(y, _x) + D * numpy.matmul(_y, _x)
            )

def Plasticity_ABCDE(inputs, outputs, A, B, C, D, E):
    x = numpy.expand_dims(inputs, axis=0)
    y = numpy.expand_dims(outputs, axis=1) 
    _x = numpy.ones_like(x)
    _y = numpy.ones_like(y)

    return (A * numpy.matmul(y, x) + B * numpy.matmul(_y, x) 
            + C * numpy.matmul(y, _x) + D * numpy.matmul(_y, _x)
            + E * numpy.matmul(x, y))

def Plasticity_SABCDE(inputs, outputs, S, A, B, C, D, E):
    x = numpy.expand_dims(inputs, axis=0)
    y = numpy.expand_dims(outputs, axis=1) 
    _x = numpy.ones_like(x)
    _y = numpy.ones_like(y)
    m = numpy.expand_dims(S, axis=1)
    m_x = x * m
    m_y = y * m

    return (A * numpy.matmul(m_y, x) + B * numpy.matmul(m, x) 
            + C * numpy.matmul(m_y, _x) + D * numpy.matmul(m, _x)
            + E * numpy.matmul(m_x, y))
