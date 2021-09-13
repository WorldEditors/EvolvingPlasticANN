"""
Layers Configuration
"""
import numpy
from activation import ActFunc
from plasticity_rule import Plasticity_ABCD, Plasticity_SABCD

class Layers(object):
    def __init__(self, 
            output_dim,
            param_name="",  
            input_keys=None, 
            output_keys=None, 
            act_type="none", 
            init_scale=0.1, 
            evolution_noise_scale=0.01, 
            pl_dict=None):
        self._name = param_name
        self._output_dim = output_dim
        self._input_keys = input_keys
        self._output_keys = output_keys
        self._act_func = ActFunc(act_type)
        self._init_scale = init_scale
        self._evolution_noise_scale = evolution_noise_scale
        self._pl_dict = pl_dict

    def initialize_layer(self, model):
        pass

    def _update_hidden(self, model, heb_is_on):
        pass

    def __call__(self, model, heb_is_on):
        """
        heb_is_on: switches for doing plastic updates
        """
        self.forward(model, heb_is_on)
        if(self._output_keys is not None):
            if(isinstance(self._output_keys, str)):
                model.get_layer(self._output_keys).set(self._hidden)
            elif(isinstance(self._output_keys, list)):
                for key in self._output_keys:
                    model.get_layer(key).set(self._hidden)
            else:
                raise Exception("Output keys must be string or list")
        return self._hidden

    # reset initial memories, only work for memory layers
    def reset(self, model):
        pass

    # set value to memories
    def set(self, value):
        assert numpy.shape(value)[-1] == self._output_dim, "in %s, set shape: %s != pre-defined shape %s"%(self._name, numpy.shape(value), self._output_dim)
        self._hidden = value

    def forward(self, model, heb_is_on):
        raise NotImplementedError("Forward not implemented")

    def backward_batch(self, grad, model):
        """
        Notice backward_batch can only be performed after forward_batch
        """
        raise NotImplementedError("Back Propagation Not Implemented")

    def forward_batch(self, model):
        raise NotImplementedError("Batch Forward Not Implemented")

    @property
    def hidden(self):
        return self._hidden

    @property
    def hidden_dim(self):
        return self._output_dim

class Input(Layers):
    """Layers for Input and Output"""
    def forward(self, model, heb_is_on):
        pass

    def backward_batch(self, grad, model):
        pass

    def forward_batch(self, model):
        pass

class FC(Layers):
    def initialize_layer(self, model):
        assert isinstance(self._input_keys, str), "input to fc layer must be a single layer"
        self._input_dim = model.get_layer(self._input_keys).hidden_dim
        if(not model.has_parameter("W_" + self._name)):
            model.add_parameter("W_" + self._name, 
                    numpy.random.normal(size=(self._output_dim, self._input_dim), loc=0.0, scale=self._init_scale), 
                    self._evolution_noise_scale)
        if(not model.has_parameter("b_" + self._name)):
            model.add_parameter("b_" + self._name, 
                    numpy.zeros(shape=(self._output_dim,), dtype="float32"), 
                    self._evolution_noise_scale)
        return

    def forward(self, model, heb_is_on):
        input_layer = model.get_layer_hidden(self._input_keys)
        self._hidden = self._act_func(numpy.matmul(model.get_parameter("W_" + self._name), input_layer) + model.get_parameter("b_" + self._name))
        if(heb_is_on and self._pl_dict is not None):
            i_s = 0
            i_e = input_layer.shape[0]
            if("input_start" in self._pl_dict):
                i_s = self._pl_dict["input_start"]
            if("input_end" in self._pl_dict):
                i_e = self._pl_dict["input_end"]

            if(self._pl_dict["type"] == "SABCD"):
                model.get_parameter("W_" + self._name)[:, i_s:i_e] += Plasticity_SABCD(
                        input_layer[i_s:i_e], self._hidden,
                        model.get_layer_hidden(self._pl_dict["S"]),
                        model.get_layer_hidden(self._pl_dict["A"]),
                        model.get_layer_hidden(self._pl_dict["B"]),
                        model.get_layer_hidden(self._pl_dict["C"]),
                        model.get_layer_hidden(self._pl_dict["D"]),
                        )
            elif(self._pl_dict["type"] == "ABCD"):
                model.get_parameter("W_" + self._name)[:, i_s:i_e] += Plasticity_ABCD(
                        input_layer[i_s:i_e], self._hidden,
                        model.get_layer_hidden(self._pl_dict["A"]),
                        model.get_layer_hidden(self._pl_dict["B"]),
                        model.get_layer_hidden(self._pl_dict["C"]),
                        model.get_layer_hidden(self._pl_dict["D"]),
                        )
            else:
                raise Exception("Unsupported plastic rule: %s"%self._pl_dict["type"])

        return self._hidden

    def forward_batch(self, model):
        input_layer = model.get_layer_hidden(self._input_keys)
        self._hidden = self._act_func(numpy.matmul(input_layer, numpy.transpose(model.get_parameter("W_" + self._name))) + model.get_parameter("b_" + self._name))
        return self._hidden

    def backward_batch(self, grads, model):
        input_layer = model.get_layer_hidden(self._input_keys)
        batch_size = input_layer.shape[0]
        g_y = grads * self._act_func.grad(self._hidden)
        g_x = numpy.matmul(g_y, model.get_parameter("W_" + self._name))
        grad_w = (1.0 / batch_size) * numpy.matmul(numpy.transpose(g_y), input_layer)
        grad_b = numpy.mean(g_y, axis=0)
        model.mod_parameter("W_" + self._name, - grad_w)
        model.mod_parameter("b_" + self._name, - grad_b)
        model.get_layer(self._input_keys).backward_batch(g_x, model)

class RandomFC(Layers):
    def initialize_layer(self, model):
        assert isinstance(self._input_keys, str), "input to fc layer must be a single layer"
        self._input_dim = model.get_layer_hidden(self._input_keys)
        if(not model.has_parameter("W_" + self._name)):
            model.add_parameter("W_" + self._name, 
                    numpy.random.normal(size=(self._output_dim, self._input_dim), loc=0.0, scale=self._init_scale), 
                    self._evolution_noise_scale)
        if(not model.has_parameter("b_" + self._name)):
            model.add_parameter("b_" + self._name, 
                    numpy.zeros(shape=(self._output_dim,), dtype="float32"), 
                    self._evolution_noise_scale)
        return

    def reset(self, model):
        self._param_w = numpy.random.normal(
                size=(self._output_dim, self._input_dim), loc=0.0, scale=self._init_scale)

    def forward(self, model, heb_is_on):
        input_layer = model.get_layer_hidden(self._input_keys)
        self._hidden = self._act_func(numpy.matmul(model.get_parameter("W_" + self._name), input_layer) + model.get_parameters("b_" + self._name))
        if(heb_is_on and self._pl_dict is not None):
            i_s = 0
            i_e = input_layer.shape[0]
            if("input_start" in self._pl_dict):
                i_s = self._pl_dict["input_start"]
            if("input_end" in self._pl_dict):
                i_e = self._pl_dict["input_end"]

            if(self._pl_dict["type"] == "SABCD"):
                model.get_parameter("W_" + self._name)[:, i_s:i_e] += Plasticity_SABCD(
                        input_layer[i_s:i_e], self._hidden,
                        model.get_layer_hidden(self._pl_dict["S"]),
                        model.get_layer_hidden(self._pl_dict["A"]),
                        model.get_layer_hidden(self._pl_dict["B"]),
                        model.get_layer_hidden(self._pl_dict["C"]),
                        model.get_layer_hidden(self._pl_dict["D"]),
                        )
            elif(self._pl_dict["type"] == "ABCD"):
                model.get_parameter("W_" + self._name)[:, i_s:i_e] += Plasticity_ABCD(
                        input_layer[i_s:i_e], self._hidden,
                        model.get_layer_hidden(self._pl_dict["A"]),
                        model.get_layer_hidden(self._pl_dict["B"]),
                        model.get_layer_hidden(self._pl_dict["C"]),
                        model.get_layer_hidden(self._pl_dict["D"]),
                        )
            else:
                raise Exception("Unsupported plastic rule: %s"%self._pl_dict["type"])

        return self._hidden

class TensorEmb(Layers):
    def initialize_layer(self, model):
        if(not model.has_parameter("b_" + self._name)):
            model.add_parameter("b_" + self._name, 
                    numpy.zeros(shape=self._output_dim, dtype="float32"), 
                    self._evolution_noise_scale)
        return

    def forward(self, model, heb_is_on):
        self._hidden = numpy.copy(model.get_parameter("b_" + self._name))
        return self._hidden

class Emb(Layers):
    def initialize_layer(self, model):
        if(not model.has_parameter("b_" + self._name)):
            model.add_parameter("b_" + self._name, 
                    numpy.zeros(shape=(self._output_dim,), dtype="float32"), 
                    self._evolution_noise_scale)
        return

    def forward(self, model, heb_is_on):
        self._hidden = numpy.copy(model.get_parameter("b_" + self._name))
        return self._hidden


class Mem(Layers):
    def initialize_layer(self, model):
        if(not model.has_parameter("H_" + self._name)):
            model.add_parameter("H_" + self._name, 
                    numpy.zeros(shape=(self._output_dim,), dtype="float32"), 
                    self._evolution_noise_scale)
        self.reset(model)
        return

    def forward(self, model, heb_is_on):
        return self._hidden

    def reset(self, model):
        self._hidden = numpy.copy(model.get_parameter("H_" + self._name))

    def set(self, value):
        assert numpy.shape(value)[-1] == self._output_dim, "Use a different sized value to set a memory: %s"%self._name
        self._hidden = numpy.copy(value)

class RandomMem(Layers):
    def initialize_layer(self, model):
        self.reset(model)
        return

    def forward(self, model, heb_is_on):
        return self._hidden

    def reset(self):
        self._hidden = numpy.random.normal(
                    size=(self._output_dim,), loc=0.0, scale=self._init_scale)

    def set(self, value):
        assert numpy.shape(value) == numpy.shape(self._hidden), "Use a different sized value to set a memory: %s"%self._name
        self._hidden = numpy.copy(value)

class EleMul(Layers):
    def initialize_layer(self, model):
        assert isinstance(self._input_keys, list), "input to EleMul layer must be a list"
        assert len(self._input_keys) == 2, "input to EleMul must be two layers"

    def forward(self, model, heb_is_on):
        self._hidden = model.get_layer_hidden(self._input_keys[0]) *model.get_layer_hidden(self._input_keys[1]) 
        return self._hidden

    def forward_batch(self, model, heb_is_on):
        self._hidden = model.get_layer_hidden(self._input_keys[0]) * model.get_layer_hidden(self._input_keys[1]) 
        return self._hidden

    def backward_batch(self, grads, model):
        model.get_layer(self._input_keys[0]).backward_batch(grads * model.get_layer_hidden(self._input_keys[1]), model)
        model.get_layer(self._input_keys[1]).backward_batch(grads * model.get_layer_hidden(self._input_keys[0]), model)

class Concat(Layers):
    def initialize_layer(self, model):
        assert isinstance(self._input_keys, list), "input to Concat layer must be a list"
    
    def forward(self, model, heb_is_on):
        concat_list = []
        for key in self._input_keys:
            concat_list.append(model.get_layer_hidden(key))
        self._hidden = numpy.concatenate(concat_list, axis=0)
        assert numpy.shape(self._hidden)[-1] == self._output_dim, "in %s, actual output shape: %s != pre-defined shape %s"%(self._name, numpy.shape(self._hidden), self._output_dim)
        return self._hidden

    def forward_batch(self, model):
        concat_list = []
        for key in self._input_keys:
            concat_list.append(model.get_layer_hidden(key))
        self._hidden = numpy.concatenate(concat_list, axis=1)
        assert numpy.shape(self._hidden)[-1] == self._output_dim, "in %s, actual output shape: %s != pre-defined shape %s"%(self._name, numpy.shape(self._hidden), self._output_dim)
        return self._hidden

    def backward_batch(self, grads, model):
        cur_b = 0
        for key in self._input_keys:
            cur_e = cur_b + model.get_layer(key).hidden_dim.shape[1]
            model.get_layer(key).backward_batch(grads[:, cur_b:cur_e], model)

class Switcher(Layers):
    """
    Give Layers of the following form: y = x[0] * x[1] + (1 - x[0]) * x[2]
    """
    def initialize_layer(self, model):
        assert isinstance(self._input_keys, list), "input to Switcher layer must be a list"
        assert len(self._input_keys) == 3, "input to Switcher must be 3 layers"

    def forward(self, model, heb_is_on):
        self._hidden = (model.get_layer_hidden(self._input_keys[0]) * model.get_layer_hidden(self._input_keys[1]) 
                + (1.0 - model.get_layer_hidden(self._input_keys[0])) * model.get_layer_hidden(self._input_keys[2]))
        assert numpy.shape(self._hidden) == self._output_dim, "in %s, actual output shape: %s != pre-defined shape %s"%(
                self._name, numpy.shape(self._hidden), self._output_dim)
        return self._hidden

    def forward_batch(self, model, heb_is_on):
        self._hidden = (model.get_layer_hidden(self._input_keys[0]) * model.get_layer_hidden(self._input_keys[1]) 
                + (1.0 - model.get_layer_hidden(self._input_keys[0])) * model.get_layer_hidden(self._input_keys[2]))
        return self._hidden

    def backward_batch(self, grads, model):
        model.get_layer(self._input_keys[0]).backward_batch(grads * (model.get_layer_hidden(self._input_keys[1]) - model.get_layer_hidden(self._input_keys[2])), model)
        model.get_layer(self._input_keys[1]).backward_batch(grads * model.get_layer_hidden(self._input_keys[0]), model)
        model.get_layer(self._input_keys[2]).backward_batch(grads * (1.0 - model.get_layer_hidden(self._input_keys[0])), model)

class SumPooling(Layers):
    """
    Summation of layers
    """
    def initialize_layer(self, model):
        assert isinstance(self._input_keys, list), "input to PoolSum layer must be a list"
    
    def forward(self, model, heb_is_on):
        self._hidden = numpy.copy(model.get_layer_hidden(self._input_keys[0]))
        for key in self._input_keys[1:]:
            self._hidden += model.get_layer_hidden(key)
        return self._hidden

    def forward_batch(self, model):
        self._hidden = numpy.copy(model.get_layer_hidden(self._input_keys[0]))
        for key in self._input_keys[1:]:
            self._hidden += model.get_layer_hidden(key)
        return self._hidden

    def backward_batch(self, grads, model):
        for key in self._input_keys:
            model.get_layer(key).backward_batch(grads, model)

class ActLayer(Layers):
    def initialize_layer(self, model):
        assert isinstance(self._input_keys, str), "input to Act Layer must be a single layer"
        self._input_dim = model.get_layer(self._input_keys).hidden_dim

    def forward(self, model, heb_is_on):
        input_layer = model.get_layer_hidden(self._input_keys)
        self._hidden = self._act_func(input_layer)
        return self._hidden

    def forward_batch(self, model):
        input_layer = model.get_layer_hidden(self._input_keys)
        self._hidden = self._act_func(input_layer)
        return self._hidden

    def backward_batch(self, grads, model):
        input_layer = model.get_layer_hidden(self._input_keys)
        g_x = grads * self._act_func.grad(self._hidden)
        model.get_layer(self._input_keys).backward_batch(g_x, model)
