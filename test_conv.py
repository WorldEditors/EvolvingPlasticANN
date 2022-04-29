import numpy
from numpy.lib.stride_tricks import as_strided

def conv2d(a, f, stride):
    w_out = (a.shape[0] - f.shape[1]) / stride[0] + 1
    h_out = (a.shape[1] - f.shape[2]) / stride[1] + 1
    s = (int(w_out), int(h_out)) + f.shape
    tmp_strides = (int(a.strides[0] * stride[0]), int(a.strides[1] * stride[1]), a.strides[2], a.strides[0], a.strides[1])
    print(s, tmp_strides)
    subM = as_strided(a, shape = s, strides=tmp_strides)
    print(subM)
    print(subM.shape)
    return numpy.einsum('ijklm,klm->ij', subM, f) 


img = numpy.zeros(shape=(5, 5, 2))
img[0,0,1]=1
img[1,1,1]=2
img[2,2,1]=3
img[1,3,0]=64
img[3,3,1]=4
kernel = numpy.zeros(shape=(2, 3, 3))
kernel[0,1,1] = 0.1
kernel[1,1,1] = 0.9
print(conv2d(img, kernel,[2,2]))
