"""
Layers Configuration
"""
import numpy
import time
from numpy.lib.stride_tricks import as_strided
from epann.parameters import Parameters
from epann.utils import categorical
from epann.activation import ActFunc

class Layers(object):
    def __init__(self, **kw_args):
        if("param_init_scale" not in kw_args):
            kw_args["param_init_scale"] = 0.10
        if("dtype" not in kw_args):
            kw_args["dtype"] = "float32"
        if("param_name_prefix" not in kw_args):
            raise Exception("Require arg param_name_prefix")
        if("params" not in kw_args):
            raise Exception("Require arg params")
        if("output_shape" not in kw_args):
            raise Exception("Require arg output_shape")
        for key in kw_args:
            self.__dict__[key] = kw_args[key]
        if("activation" in kw_args):
            self._act_func = ActFunc(self.activation)

    def add_parameter(self, shape, name):
        self.params.add(self.param_name_prefix + "/" + name, 
                    numpy.random.normal(size=shape, loc=0.0, scale=self.param_init_scale).astype(self.dtype))

    def parameter(self, name):
        return self.params.get(self.param_name_prefix + "/" + name)

    def __call__(self, inputs=None, **kw_args):
        """
        heb_is_on: switches for doing plastic updates
        """
        if(inputs is not None):
            assert numpy.shape(inputs) == self.input_shape, \
                    "input shape of layer %s do not match, required %s, received %s"%(
                            self.param_name_prefix, self.input_shape, numpy.shape(inputs))
        outputs = self.forward(inputs, **kw_args)
        if(outputs is not None):
            assert numpy.shape(outputs) == self.output_shape, \
                    "output shape of layer %s do not match, required %s, received %s"%(
                            self.param_name_prefix, self.output_shape, numpy.shape(outputs))
        return outputs

    # set value to memories
    def reset(self, **kw_args):
        """
        Typically used to reset memories
        """
        raise NotImplementedError("Layer reset not implemented")

    def forward(self, inputs=None, **kw_args):
        raise NotImplementedError("Layer forward not implemented")

    def backward_batch(self, grads, inputs=None, **kw_args):
        """
        Notice backward_batch can only be performed after forward_batch
        """
        raise NotImplementedError("Back propagation not implemented")

    def forward_batch(self, inputs=None, **kw_args):
        raise NotImplementedError("Batch Forward Not Implemented")

class Memory(Layers):
    """
    Read and writable memory with trainable initial parameters
    """
    def __init__(self, **kw_args):
        if("writable" not in kw_args):
            kw_args["writable"] = False
        if("initialize_settings" not in kw_args):
            #Only 'P', 'R', 'C' is available
            #'P': reset to some parameters
            #'R': reset to some random parameters
            #'C': reset to some constant
            kw_args["initialize_settings"] = 'P'
        if(kw_args["initialize_settings"] != 'P'):
            if("initialize_hyper_parameter" not in kw_args):
                kw_args["initialize_hyper_parameter"] = 0.0

        super(Memory, self).__init__(**kw_args)

        self.input_shape = self.output_shape
        
        if(self.initialize_settings == 'P'):
            self.add_parameter(self.output_shape, "Memory")

    def forward(self, inputs=None, **kw_args):
        if(inputs is not None):
            if(not self.writable):
                raise Exception("Trying to write a non-writable memory")
            self.memory = numpy.copy(inputs)
        return numpy.copy(self.memory)

    # set value to memories
    def reset(self, **kw_args):
        if(self.initialize_settings == 'P'):
            self.memory = numpy.copy(self.parameter("Memory"))
        elif(self.initialize_settings == 'R'):
            self.memory = numpy.random.normal(size=self.output_shape, loc=0.0, scale=self.initialize_hyper_parameter)
        elif(self.initialize_settings == 'C'):
            self.memory = numpy.full(shape=self.output_shape, fill_value=self.initialize_hyper_parameter)
        else:
            raise Exception("illegal intialize_settings, only P/R/C is permitted")
        self.memory = self.memory.astype(self.dtype) 

class FIFOMemory(Layers):
    """
    Read and writable memory with trainable initial parameters
    """
    def __init__(self, **kw_args):
        if("writable" not in kw_args):
            kw_args["writable"] = False
        if("initialize_hyper_parameter" not in kw_args):
            kw_args["initialize_hyper_parameter"] = 0.0
        super(FIFOMemory, self).__init__(**kw_args)
        assert len(self.output_shape)==2, "FIFOMemory can support only 2-dimensional outputs"

        self.input_shape = (self.output_shape[1], )
        self.reset()
        
    def forward(self, inputs=None, **kw_args):
        if(inputs is not None):
            if(not self.writable):
                raise Exception("Trying to write a non-writable memory")
            self.memory = numpy.roll(self.memory, -1, axis=0)
            self.memory[-1] = numpy.copy(inputs)
        return numpy.copy(self.memory)

    # set value to memories
    def reset(self, **kw_args):
        self.memory = numpy.full(shape=self.output_shape, fill_value=self.initialize_hyper_parameter)

class FC(Layers):
    def __init__(self, **kw_args):
        if("activation" not in kw_args):
            kw_args["activation"] = "relu"
        if("plastic" not in kw_args):
            kw_args["plastic"] = False
        if("bias" not in kw_args):
            kw_args["bias"] = True
        if("input_shape" not in kw_args):
            raise Exception("Require arg input_shape")

        super(FC, self).__init__(**kw_args)
        assert len(self.input_shape) == len(self.output_shape)
        if(len(self.input_shape) > 1):
            print("input dimension > 1, the batch of inputs must be on dimension >= 1")
            bias_shape = self.output_shape[:1] + (1,) * (len(self.output_shape) - 1)
        else:
            bias_shape = self.output_shape[:1]
        if(not self.plastic):
            self.add_parameter((self.output_shape[0], self.input_shape[0]), "W")
        if(self.bias):
            self.add_parameter(bias_shape, "b")

    def forward(self, inputs, **kw_args):
        if(self.plastic):
            if("weight" not in kw_args):
                raise Exception("In plastic mode, weight must be given in the args")
            else:
                w = kw_args["weight"]
        else:
            w = self.parameter("W")
        if(self.bias):
            outputs = self._act_func(numpy.matmul(w, inputs)  + self.parameter("b"))
        else:
            outputs = self._act_func(numpy.matmul(w, inputs))

        return outputs

    def reset(self, **kw_args):
        pass

class Conv(Layers):
    """
    Convolutional Layers
    """
    def __init__(self, **kw_args):
        if("input_shape" not in kw_args):
            raise Exception("Require arg input_shape")
        if("kernel_shape" not in kw_args):
            raise Exception("Require arg kernel_shape")
        if("stride" not in kw_args):
            kw_args["stride"] = 1
        super(Conv, self).__init__(**kw_args)

        assert self.kernel_shape[0] % 2 == 1 and self.kernel_shape[1] % 2 == 1, "kernel size must be odd"
        self._W, self._H, self._C = self.input_shape
        self.w_out = (self._W - self.kernel_shape[0]) // self.stride + 1
        self.h_out = (self._H - self.kernel_shape[1]) // self.stride + 1
        assert self.output_shape == (self.w_out, self.h_out, self.output_shape[2]), "output shape of convolution layer do not match what is specified"
        self.add_parameter((self.output_shape[2], self.kernel_shape[0], self.kernel_shape[1], self._C),
            "kernel")

    def forward(self, inputs, **kw_args):
        tmp_shape = (self.w_out, self.h_out) + self.kernel_shape + (self._C,)
        strd = inputs.strides
        tmp_strides = (strd[0] * self.stride, strd[1] * self.stride, strd[0], strd[1], strd[2])
        tmp_img = as_strided(inputs, shape=tmp_shape, strides=tmp_strides)
        outputs = numpy.einsum('ijklm,nklm->ijn', tmp_img, self.parameter("kernel"))
        return outputs

    def reset(self, **kw_args):
        pass

class Pooling(Layers):
    # Pooling Layers
    def __init__(self, **kw_args):
        if("input_shape" not in kw_args):
            raise Exception("Require arg input_shape")
        if("kernel_shape" not in kw_args):
            raise Exception("Require arg kernel_shape")
        if("pooling_type" not in kw_args):
            # Mean, Max, Sum Pooling
            kw_args["pooling_type"] = "Mean"
        if("stride" not in kw_args):
            kw_args["stride"] = 1
        super(Pooling, self).__init__(**kw_args)

        assert self.kernel_shape[0] % 2 == 1 and self.kernel_shape[1] % 2 == 1, "kernel size must be odd"
        self._W, self._H, self._C = self.input_shape
        self.w_out = (self._W - self.kernel_shape[0]) // self.stride + 1
        self.h_out = (self._H - self.kernel_shape[1]) // self.stride + 1
        assert self.output_shape == (self.w_out, self.h_out, self.output_shape[2]), "output shape of convolution layer do not match what is specified"

    def forward(self, inputs, **kw_args):
        tmp_shape = (self.w_out, self.h_out, self._C) + self.kernel_shape
        strd = inputs.strides
        tmp_strides = (strd[0] * self.stride, strd[1] * self.stride, strd[2], strd[0], strd[1])
        tmp_img = as_strided(inputs, shape=tmp_shape, strides=tmp_strides)
        if(self.pooling_type=="Mean"):
            outputs = numpy.mean(tmp_img, axis=(3, 4))
        if(self.pooling_type=="Max"):
            outputs = numpy.max(tmp_img, axis=(3, 4))
        if(self.pooling_type=="Sum"):
            outputs = numpy.sum(tmp_img, axis=(3, 4))
        return outputs

    def reset(self, **kw_args):
        pass

class Hebbian(Memory):
    def __init__(self, **kw_args):
        kw_args["writable"] = True
        super(Hebbian, self).__init__(**kw_args)

        self.add_parameter(self.output_shape, "A")
        self.add_parameter(self.output_shape, "B")
        self.add_parameter(self.output_shape, "C")
        self.add_parameter(self.output_shape, "D")

    def forward(self, inputs=None, **kw_args):
        if("pre_syn" not in kw_args or "post_syn" not in kw_args):
            return numpy.copy(self.memory)
        pre_syn = kw_args["pre_syn"]
        post_syn = kw_args["post_syn"]
        assert((post_syn.shape[0], pre_syn.shape[0]) == self.output_shape and len(post_syn.shape)==1 and len(pre_syn.shape)==1),\
                "the pre-synaptic and post-synaptic dimensions do not match"
        post_unit = numpy.ones_like(post_syn)
        pre_unit = numpy.ones_like(pre_syn)
        deta_whts = 0.05 * (numpy.outer(post_syn, pre_syn) * self.parameter("A") 
                + numpy.outer(post_syn, pre_unit) * self.parameter("B")
                + numpy.outer(post_unit, pre_syn) * self.parameter("C")
                + numpy.outer(post_unit, pre_unit) * self.parameter("D"))

        if("modulator" in kw_args):
            mod = kw_args["modulator"]
            if(mod.shape==(self.output_shape[0], )):
                pre_unit = numpy.ones_like(pre_syn)
                assert mod.shape == post_syn.shape, "the shape of modulator and post synaptic signals do not match"
                self.memory += numpy.outer(mod, pre_unit) * deta_whts
            elif(numpy.product(mod.shape)==1):
                self.memory += mod * deta_whts
            else:
                raise Exception("modulator shape illegal")
        else:
            self.memory += deta_whts 

        return numpy.copy(self.memory)

    # set value to memories
    def reset(self, **kw_args):
        super(Hebbian, self).reset(**kw_args)

class Hebbian2(Memory):
    def __init__(self, **kw_args):
        kw_args["writable"] = True
        if("static" not in kw_args):
            kw_args["static"] = False

        super(Hebbian2, self).__init__(**kw_args)

        self.evo_path = Memory(params=self.params, 
                param_name_prefix=self.param_name_prefix + "/evo_path", 
                output_shape=self.output_shape,
                writable="True", 
                initialize_settings='C')
        self.add_parameter((1,), "\eta")

    def forward(self, inputs=None, **kw_args):
        if("pre_syn" not in kw_args or "post_syn" not in kw_args or self.static):
            return numpy.copy(self.memory)

        pre_syn = kw_args["pre_syn"]
        post_syn = kw_args["post_syn"]
        assert((post_syn.shape[0], pre_syn.shape[0]) == self.output_shape and len(post_syn.shape)==1 and len(pre_syn.shape)==1),\
                "the pre-synaptic and post-synaptic dimensions do not match"
        whts = self.evo_path()
        eta = 1.0#self.parameter("\eta") + 10.0
        #eta = 0.50 * eta / (numpy.abs(eta) + 1) + 0.50
        alpha = 0.2
        n_whts = eta * numpy.outer(post_syn, pre_syn) + (1 - eta) * whts

        if("modulator" in kw_args):
            mod = kw_args["modulator"]
            if(mod.shape==(self.output_shape[0], )):
                pre_unit = numpy.ones_like(pre_syn)
                assert mod.shape == post_syn.shape, "the shape of modulator and post synaptic signals do not match"
                self.memory += alpha * numpy.outer(mod, pre_unit) * n_whts
            elif(numpy.product(mod.shape)==1):
                self.memory += alpha * kw_args["modulator"] * n_whts
            else:
                raise Exception("modulator shape illegal")
        else:
            self.memory += alpha * n_whts 
        self.evo_path(n_whts)

        return numpy.copy(self.memory)

    # set value to memories
    def reset(self, **kw_args):
        super(Hebbian2, self).reset(**kw_args)
        self.evo_path.reset()

class Hebbian3(Memory):
    def __init__(self, **kw_args):
        kw_args["writable"] = True

        super(Hebbian3, self).__init__(**kw_args)
        assert len(self.output_shape) == 2
        self.d_y, self.d_x = self.output_shape

        #self.evo_path = Memory(params=self.params, 
        #        param_name_prefix=self.param_name_prefix + "/evo_path", 
        #        output_shape=self.output_shape,
        #        writable="True", 
        #        initialize_settings='C')
        self.add_parameter((self.d_x,), "/W_Ax")
        self.add_parameter((self.d_y,), "/W_Ay")
        self.add_parameter((self.d_x,), "/W_Bx")
        self.add_parameter((self.d_y,), "/W_By")
        self.add_parameter((self.d_x,), "/W_Cx")
        self.add_parameter((self.d_y,), "/W_Cy")
        self.add_parameter((self.d_x,), "/W_Dx")
        self.add_parameter((self.d_y,), "/W_Dy")
        #self.add_parameter((1,), "\eta")

    def forward(self, inputs=None, **kw_args):
        if("pre_syn" not in kw_args or "post_syn" not in kw_args):
            return numpy.copy(self.memory)
        pre_syn = kw_args["pre_syn"]
        post_syn = kw_args["post_syn"]
        assert((post_syn.shape[0], pre_syn.shape[0]) == self.output_shape and len(post_syn.shape)==1 and len(pre_syn.shape)==1),\
                "the pre-synaptic and post-synaptic dimensions do not match"
        post_unit = numpy.ones_like(post_syn)
        pre_unit = numpy.ones_like(pre_syn)
        n_whts = (numpy.outer(post_syn, pre_syn) * self.p_a
                + numpy.outer(post_syn, pre_unit) * self.p_b
                + numpy.outer(post_unit, pre_syn) * self.p_c
                + numpy.outer(post_unit, pre_unit) * self.p_d)

        #whts = self.evo_path()
        #eta = self.parameter("\eta")
        #n_whts = eta * deta + (1 - eta) * whts

        if("modulator" in kw_args):
            mod = kw_args["modulator"]
            if(mod.shape==(self.output_shape[0], )):
                f_mod = numpy.repeat(numpy.expand(mod, axis=1), repeats=self.d_x, axis=1)
                self.memory += 0.20 * f_mod * n_whts
            elif(numpy.product(mod.shape)==1):
                self.memory += 0.20 * mod * n_whts
            else:
                raise Exception("modulator shape illegal")
        else:
            self.memory += n_whts 

        return numpy.copy(self.memory)

    # set value to memories
    def reset(self, **kw_args):
        super(Hebbian3, self).reset(**kw_args)
        #self.evo_path.reset()
        self.p_a = numpy.outer(self.parameter("/W_Ay"), self.parameter("/W_Ax"))
        self.p_b = numpy.outer(self.parameter("/W_By"), self.parameter("/W_Bx"))
        self.p_c = numpy.outer(self.parameter("/W_Cy"), self.parameter("/W_Cx"))
        self.p_d = numpy.outer(self.parameter("/W_Dy"), self.parameter("/W_Dx"))

class PlasticFC(Layers):
    def __init__(self, **kw_args):
        if("activation" not in kw_args):
            kw_args["activation"] = "tanh"
        if("hebbian_type" not in kw_args):
            kw_args["hebbian_type"] = 3
        if("initialize_settings" not in kw_args):
            kw_args["initialize_settings"] = 'P'
        super(PlasticFC, self).__init__(**kw_args)

        if(self.hebbian_type == 1):
            self.heb = Hebbian(params=self.params, 
                    param_name_prefix=self.param_name_prefix + "/hebbian",
                    output_shape=(self.output_shape[0], self.input_shape[0]),
                    initialize_settings=self.initialize_settings,
                    param_init_scale=self.param_init_scale,
                    )
        elif(self.hebbian_type == 2):
            self.heb = Hebbian2(params=self.params, 
                    param_name_prefix=self.param_name_prefix + "/hebbian",
                    output_shape=(self.output_shape[0], self.input_shape[0]),
                    initialize_settings=self.initialize_settings,
                    param_init_scale=self.param_init_scale,
                    )
        elif(self.hebbian_type == 3):
            self.heb = Hebbian3(params=self.params, 
                    param_name_prefix=self.param_name_prefix + "/hebbian",
                    output_shape=(self.output_shape[0], self.input_shape[0]),
                    initialize_settings=self.initialize_settings,
                    param_init_scale=self.param_init_scale,
                    )
        self.fc = FC(params=self.params, 
                param_name_prefix=self.param_name_prefix + "/fc",
                output_shape=self.output_shape, 
                input_shape=self.input_shape, 
                activation=self.activation, 
                param_init_scale=self.param_init_scale,
                plastic=True)
        self.act_func = ActFunc(self.activation)
        self.ready_to_learn = False

    def forward(self, inputs=None, **kw_args):
        assert inputs is not None, "inputs to a PlastiRecursion layer can not be None"
        self.pre_syn = inputs
        self.post_syn = self.fc(self.pre_syn, weight=self.heb())
        self.ready_to_learn = True

        return self.post_syn
    
    def reset(self, **kw_args):
        self.heb.reset()

    def learn(self, **kw_args):
        if(not self.ready_to_learn):
            raise Exception("Must call forward before learn")
        if("modulator" not in kw_args or kw_args["modulator"] is None):
            self.heb(pre_syn=self.pre_syn, post_syn=self.post_syn)
        else:
            self.heb(pre_syn=self.pre_syn, post_syn=self.post_syn, modulator=kw_args["modulator"])
        self.ready_to_learn = False

class PlasticUnFC(Layers):
    def __init__(self, **kw_args):
        if("activation" not in kw_args):
            kw_args["activation"] = "tanh"
        if("initialize_settings" not in kw_args):
            kw_args["initialize_settings"] = 'P'
        if("inner_psize" not in kw_args):
            kw_args["inner_psize"] = 16
        super(PlasticUnFC, self).__init__(**kw_args)

        self.heb = Memory(params=self.params, 
                param_name_prefix=self.param_name_prefix + "/weights",
                output_shape=(self.output_shape[0], self.input_shape[0]),
                initialize_settings=self.initialize_settings,
                param_init_scale=self.param_init_scale,
                writable=True
                )
        self.fc = FC(params=self.params, 
                param_name_prefix=self.param_name_prefix + "/fc",
                output_shape=self.output_shape, 
                input_shape=self.input_shape, 
                activation=self.activation,
                param_init_scale=self.param_init_scale,
                plastic=True)

        self.add_parameter((self.inner_psize, 3), "/p_w1")
        self.add_parameter((self.inner_psize, 1, 1), "/p_b1")
        self.add_parameter((1, self.inner_psize), "/p_w2")
        self.add_parameter((1, 1, 1), "/p_b2")

        self.sign_func = ActFunc("softsign")

    def forward(self, inputs=None, **kw_args):
        assert inputs is not None, "inputs to a PlastiRecursion layer can not be None"
        whts = self.heb()

        outputs = self.fc(inputs, weight=self.heb())
        post_syn = numpy.repeat(numpy.expand_dims(outputs, axis=[0,2]), self.input_shape[0], axis=2)
        pre_syn = numpy.repeat(numpy.expand_dims(inputs, axis=[0,1]), self.output_shape[0], axis=1)
        pl_concat = numpy.concatenate([numpy.expand_dims(whts, axis=[0]), post_syn, pre_syn], axis=0)
        pl = self.sign_func(numpy.einsum("ij,jkl->ikl", self.parameter("/p_w1"), pl_concat) + self.parameter("/p_b1"))
        pl = self.sign_func(numpy.einsum("ij,jkl->ikl", self.parameter("/p_w2"), pl) + self.parameter("/p_b2"))
        deta = numpy.squeeze(pl)
        self.heb(whts + deta)

        return outputs
    
    def reset(self, **kw_args):
        self.heb.reset()

class PlasticRNN(Layers):
    def __init__(self, **kw_args):
        if("activation" not in kw_args):
            kw_args["activation"] = "tanh"
        if("plasticity_type" not in kw_args):
            kw_args["plasticity_type"] = "naive"
        if("initialize_settings" not in kw_args):
            kw_args["initialize_settings"] = 'C'
        super(PlasticRNN, self).__init__(**kw_args)

        if(self.hebbian_type == 1):
            self.heb_h = Hebbian(params=self.params, 
                    param_name_prefix=self.param_name_prefix + "/hebbian_h",
                    output_shape=(self.output_shape[0], self.output_shape[0]),
                    initialize_settings=self.initialize_settings,
                    param_init_scale=self.param_init_scale)
            self.heb_x = Hebbian(params=self.params, 
                    param_name_prefix=self.param_name_prefix + "/hebbian_x",
                    output_shape=(self.output_shape[0], self.input_shape[0]),
                    initialize_settings=self.initialize_settings,
                    param_init_scale=self.param_init_scale)
        elif(self.hebbian_type == 2):
            self.heb_h = Hebbian2(params=self.params, 
                    param_name_prefix=self.param_name_prefix + "/hebbian_h",
                    output_shape=(self.output_shape[0], self.output_shape[0]),
                    initialize_settings=self.initialize_settings,
                    param_init_scale=self.param_init_scale)
            self.heb_x = Hebbian2(params=self.params, 
                    param_name_prefix=self.param_name_prefix + "/hebbian_x",
                    output_shape=(self.output_shape[0], self.input_shape[0]),
                    initialize_settings='P',
                    param_init_scale=self.param_init_scale,
                    static=True)
        elif(self.hebbian_type == 3):
            self.heb_h = Hebbian3(params=self.params, 
                    param_name_prefix=self.param_name_prefix + "/hebbian_h",
                    output_shape=(self.output_shape[0], self.output_shape[0]),
                    initialize_settings=self.initialize_settings,
                    param_init_scale=self.param_init_scale)
            self.heb_x = Hebbian3(params=self.params, 
                    param_name_prefix=self.param_name_prefix + "/hebbian_x",
                    output_shape=(self.output_shape[0], self.input_shape[0]),
                    initialize_settings=self.initialize_settings,
                    param_init_scale=self.param_init_scale)

        self.mem = Memory(params=self.params, 
                param_name_prefix=self.param_name_prefix + "/mem", 
                output_shape=(self.output_shape[0],), 
                writable="True", 
                initialize_settings='C')
        self.add_parameter(self.output_shape, "/b")
        self.act_func = ActFunc(self.activation)

    def forward(self, inputs=None, **kw_args):
        assert inputs is not None, "inputs to a PlastiRecursion layer can not be None"
        self.pre_syn_h = self.mem()
        w_h = self.heb_h()
        w_x = self.heb_x()
        self.pre_syn_x = numpy.copy(inputs)
        self.post_syn = self.act_func(numpy.matmul(w_h, self.pre_syn_h) + numpy.matmul(w_x, inputs) + self.parameter("/b"))
        self.ready_to_learn = True

        return self.mem(self.post_syn)

    def reset(self, **kw_args):
        self.heb_h.reset()
        self.heb_x.reset()
        self.mem.reset()

    def learn(self, **kw_args):
        if(not self.ready_to_learn):
            raise Exception("Must call forward before learn")
        if("modulator" not in kw_args):
            self.heb_h(pre_syn=self.pre_syn_h, post_syn=self.post_syn)
            self.heb_x(pre_syn=self.pre_syn_x, post_syn=self.post_syn)
        elif(kw_args["modulator"] is None):
            self.heb_h(pre_syn=self.pre_syn_h, post_syn=self.post_syn)
            self.heb_x(pre_syn=self.pre_syn_x, post_syn=self.post_syn)
        else:
            self.heb_h(pre_syn=self.pre_syn_h, post_syn=self.post_syn, modulator=kw_args["modulator"]["h"])
            self.heb_x(pre_syn=self.pre_syn_x, post_syn=self.post_syn, modulator=kw_args["modulator"]["x"])
        self.ready_to_learn = False

class SimpleRNN(Layers):
    def __init__(self, **kw_args):
        if("activation" not in kw_args):
            kw_args["activation"] = "tanh"
        super(SimpleRNN, self).__init__(**kw_args)

        self.mem = Memory(params=self.params, 
                param_name_prefix=self.param_name_prefix + "/mem", 
                output_shape=(self.output_shape[0],), 
                writable="True", 
                param_init_scale=self.param_init_scale,
                initialize_settings='C')
        self.fc = FC(params=self.params, 
                param_name_prefix=self.param_name_prefix + "/fc",
                output_shape=self.output_shape,
                input_shape=(self.input_shape[0] + self.output_shape[0], ),
                param_init_scale=self.param_init_scale,
                activation=self.activation)

    def forward(self, inputs=None, **kw_args):
        assert inputs is not None, "inputs to a PlastiRecursion layer can not be None"
        h_t = self.mem()
        o_t = self.fc(numpy.concatenate([inputs, h_t], axis=0))
        self.mem(o_t)
        return o_t

    def reset(self, **kw_args):
        self.mem.reset()

class LSTM(Layers):
    def __init__(self, **kw_args):
        if("activation" not in kw_args):
            kw_args["activation"] = "tanh"
        if("gate_activation" not in kw_args):
            kw_args["gate_activation"] = "sigmoid"
        super(LSTM, self).__init__(**kw_args)

        self.mem_c = Memory(params=self.params, 
                param_name_prefix=self.param_name_prefix + "/mem_c", 
                output_shape=(self.output_shape[0],), 
                writable="True", 
                initialize_settings='C')
        self.mem_h = Memory(params=self.params, 
                param_name_prefix=self.param_name_prefix + "/mem_h", 
                output_shape=(self.output_shape[0],), 
                writable="True", 
                initialize_settings='C')
        self.fc = FC(params=self.params, 
                param_name_prefix=self.param_name_prefix + "/fc",
                output_shape=(4 * self.output_shape[0],), 
                input_shape=(self.input_shape[0] + self.output_shape[0], ),
                param_init_scale=self.param_init_scale,
                activation="none") 
        self.gate_act_func = ActFunc(self.gate_activation)
        self.act_func = ActFunc(self.activation)

    def forward(self, inputs=None, **kw_args):
        assert inputs is not None, "inputs to a PlastiRecursion layer can not be None"
        h_t = self.mem_h()
        outputs = self.fc(numpy.concatenate([inputs, h_t], axis=0))
        split_outputs = numpy.split(outputs, 4)
        i_t = self.gate_act_func(split_outputs[1])
        f_t = self.gate_act_func(split_outputs[2])
        o_t = self.gate_act_func(split_outputs[3])
        c_t_n = f_t * self.mem_c() + i_t * self.act_func(split_outputs[0])
        h_t_n = o_t * self.act_func(c_t_n)
        self.mem_c(inputs=c_t_n)
        self.mem_h(inputs=h_t_n)
        return h_t_n

    def reset(self, **kw_args):
        self.mem_c.reset()
        self.mem_h.reset()

class VSML(Layers):
    def __init__(self, **kw_args):
        if("inner_size" not in kw_args):
            kw_args["inner_size"] = 8
        if("m_size" not in kw_args):
            kw_args["m_size"] = 1

        super(VSML, self).__init__(**kw_args)
        assert len(self.input_shape) == 1 and len(self.output_shape) ==1 and self.input_shape[0] % self.m_size == 0 and self.output_shape[0] % self.m_size == 0

        self.input_units = self.input_shape[0] // self.m_size
        self.output_units = self.output_shape[0] // self.m_size

        self.mem_h = Memory(params=self.params, 
                param_name_prefix=self.param_name_prefix + "/mem_h", 
                output_shape=(self.input_units, self.output_units, self.inner_size), 
                writable="True", 
                initialize_settings='C')

        self.mem_c = Memory(params=self.params, 
                param_name_prefix=self.param_name_prefix + "/mem_c", 
                output_shape=(self.input_units, self.output_units, self.inner_size), 
                writable="True", 
                initialize_settings='C')

        self.mem_n = Memory(params=self.params, 
                param_name_prefix=self.param_name_prefix + "/mem_n", 
                output_shape=(self.output_units, self.m_size), 
                writable="True", 
                initialize_settings='C')

        self.add_parameter((4 * self.inner_size, 2 * self.m_size + self.inner_size), "W")
        self.add_parameter((4 * self.inner_size, ), "b")
        self.add_parameter((self.m_size, self.inner_size), "Wo")

        self.gate_act_func = ActFunc("sigmoid")

    def forward(self, inputs=None, **kw_args):
        split_inputs = numpy.array(numpy.split(numpy.array(inputs), self.input_units))
        ext_i1 = numpy.repeat(numpy.expand_dims(split_inputs, axis=1), repeats=self.output_units, axis=1)
        ext_i2 = numpy.repeat(numpy.expand_dims(self.mem_n(), axis=0), repeats=self.input_units, axis=0)
        concat_inputs = numpy.concatenate([self.mem_h(), ext_i1, ext_i2], axis=-1)
        concat_outputs = numpy.einsum("ij,lmj->lmi", self.parameter("W"), concat_inputs) + self.parameter("b") 
        split_outputs = numpy.split(concat_outputs, 4, axis=-1)
        i_t = self.gate_act_func(split_outputs[0])
        f_t = self.gate_act_func(split_outputs[1])
        o_t = self.gate_act_func(split_outputs[2])
        g_t = numpy.tanh(split_outputs[3])
        c_t = f_t * self.mem_c() + i_t * g_t
        h_t = o_t * numpy.tanh(c_t)
        outputs = numpy.mean(h_t, axis=0)
        outputs = numpy.einsum("ij,kj->ki", self.parameter("Wo"), outputs)

        self.mem_c(c_t)
        self.mem_h(h_t)
        self.mem_n(outputs)
        
        return numpy.ravel(outputs)

    def reset(self, **kw_args):
        self.mem_h.reset()
        self.mem_c.reset()
        self.mem_n.reset()

def test_layers():
    params = Parameters()
    # Test FC
    test_layer_1 = PlasticFC(params=params, param_name_prefix="PFC", output_shape=(128,), input_shape=(64,), activation="tanh", initialize_settings='R', plasticity_type="modulated")
    test_layer_2 = SimpleRNN(params=params, param_name_prefix="RNN", output_shape=(128,), input_shape=(64,), activation="tanh", initialize_settings='P')
    test_layer_3 = PlasticRNN(params=params, param_name_prefix="PRNN", output_shape=(128,), input_shape=(64,), activation="tanh", initialize_settings='R', plasticity_type="modulated")
    test_layer_4 = LSTM(params=params, param_name_prefix="LSTM", output_shape=(128,), input_shape=(64,), activation="tanh", gate_activation="sigmoid")
    test_layer_5 = MemoryMachine(params=params, param_name_prefix="MemoryMachine", output_shape=(64,), input_shape=(64,), activation="tanh", gate_activation="sigmoid")
    test_layer_6 = VSML(params=params, param_name_prefix="VSML", output_shape=(64,), input_shape=(64,), inner_size=16, m_size=2)
    test_layer_1.reset()
    test_layer_2.reset()
    test_layer_3.reset()
    test_layer_4.reset()
    test_layer_5.reset()
    test_layer_6.reset()

    for key in params.parameters.keys():
        print(key, params.parameters[key].shape)
    param_vec, params_shape = params.to_vector
    print(param_vec.shape, params_shape)

    inputs = numpy.random.rand(1000, 64)
    t_1 = time.time()
    for i in range(inputs.shape[0]):
        outputs_1 = test_layer_1(inputs[i])
    t_2 = time.time()
    for i in range(inputs.shape[0]):
        outputs_2 = test_layer_2(inputs[i])
    t_3 = time.time()
    for i in range(inputs.shape[0]):
        outputs_3 = test_layer_3(inputs[i])
    t_4 = time.time()
    for i in range(inputs.shape[0]):
        outputs_4 = test_layer_4(inputs[i])
    t_5 = time.time()
    for i in range(inputs.shape[0]):
        outputs_5 = test_layer_5(inputs[i])
    t_6 = time.time()
    for i in range(inputs.shape[0]):
        outputs_6 = test_layer_6(inputs[i].tolist())
    t_7 = time.time()
    print(t_2 - t_1, t_3 - t_2, t_4 - t_3, t_5 - t_4, t_6 - t_5, t_7 - t_6)

    conv = Conv(params=params, param_name_prefix="conv", input_shape=(64, 64, 3), output_shape=(28, 28, 10), kernel_shape=(9, 9), stride=2)
    pool = Pooling(params=params, param_name_prefix="pool", input_shape=(28, 28, 10), output_shape=(13, 13, 10), kernel_shape=(3, 3), stride=2)
    inputs = numpy.random.rand(64, 64, 3)
    outputs_0 = conv(inputs)
    outputs_0 = pool(outputs_0)
    print(outputs_1.shape, outputs_2.shape, outputs_3.shape, outputs_4.shape, outputs_5.shape)

if __name__=="__main__":
    test_layers()
