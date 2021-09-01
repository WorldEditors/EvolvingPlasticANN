"""
Numpy Based Simple Neural Network Architectures, with Plasticity Implemented in configuration
"""
import sys
import numpy
import pickle
import json
from numpy import random
from copy import copy, deepcopy
from utils import reset_weights_axis
from plasticity_rule import Plasticity_S, Plasticity_ABCD, Plasticity_ABCDE, Plasticity_SABCD, Plasticity_SABCDE

class PlasticNN(object):
    def __init__(self, 
            input_neurons, 
            model_structures):

        self._parameter_list = dict()
        self._hidden_fire = dict()
        self._hidden_potential = dict()
        self._act_func = dict()
        self._act_func_grad = dict()
        self._noise_factor = dict()

        assert isinstance(input_neurons, int)
        self._model_structure = model_structures

        # define how to add meta parameters
        self._hidden_length = {"input": input_neurons}
        self._hidden_memory = set()

        def add_meta_key():
            if("Meta_A" not in self._parameter_list):
                self._parameter_list["Meta_A"] = numpy.asarray([0.0], dtype="float32")
                self._parameter_list["Meta_B"] = numpy.asarray([0.0], dtype="float32")
                self._parameter_list["Meta_C"] = numpy.asarray([0.0], dtype="float32")
                self._parameter_list["Meta_D"] = numpy.asarray([0.0], dtype="float32")
                self._noise_factor["Meta_A"] = 1.0e-3
                self._noise_factor["Meta_B"] = 1.0e-3
                self._noise_factor["Meta_C"] = 1.0e-3
                self._noise_factor["Meta_D"] = 1.0e-3

        def add_meta_lr():
            #Inner Loop Learning rate for evolutionary MAML; 
            self._parameter_list["Inner_LR"] = numpy.asarray([0.5, 0.2, 0.1, 0.05], dtype="float32")
            self._noise_factor["Inner_LR"] = 1.0e-3
        
        for key in self._model_structure:
            connect_type, input_layers, output_num, _, scale, noise_factor, _ = self._model_structure[key]
            self._hidden_length[key] = output_num

        for key in self._model_structure:
            connect_type, input_layers, output_num, _, scale, noise_factor, pl_rule = self._model_structure[key]
            if(pl_rule is not None):
                if(pl_rule["type"] == "S" 
                        and "Meta_A" not in self._parameter_list):
                    add_meta_key()

            input_size = 0
            if(input_layers is not None):
                for input_layer in input_layers:
                    input_size += self._hidden_length[input_layer]

            if(connect_type == "fc"):
                self._parameter_list["W_" + key] = numpy.random.normal(
                        size=(output_num, input_size), loc=0.0, scale=scale/numpy.sqrt(input_size))
                self._parameter_list["b_" + key] = numpy.zeros(shape=(output_num,), dtype="float32")
                self._noise_factor["W_" + key] = noise_factor
                self._noise_factor["b_" + key] = noise_factor
            elif(connect_type == "rnn"):
                self._hidden_memory.add(key)
                #input size is different for recursive layers
                input_size += output_num
                self._parameter_list["W_" + key] = numpy.random.normal(size=(output_num, input_size),
                        loc=0.0, scale=scale/numpy.sqrt(input_size))
                self._parameter_list["b_" + key] = numpy.zeros(shape=(output_num,), dtype="float32")
                self._parameter_list["H_Mem_" + key] = numpy.zeros(shape=(output_num,), dtype="float32")
                self._noise_factor["W_" + key] = noise_factor
                self._noise_factor["b_" + key] = noise_factor
                self._noise_factor["H_Mem_" + key] = noise_factor
            elif(connect_type == "grnn"):
                self._hidden_memory.add(key)
                input_size += output_num
                self._parameter_list["W_r_" + key] = numpy.random.normal(size=(output_num, input_size),
                        loc=0.0, scale=scale/numpy.sqrt(input_size))
                self._parameter_list["W_z_" + key] = numpy.random.normal(size=(output_num, input_size),
                        loc=0.0, scale=scale/numpy.sqrt(input_size))
                self._parameter_list["W_" + key] = numpy.random.normal(size=(output_num, input_size),
                        loc=0.0, scale=scale/numpy.sqrt(input_size))
                self._parameter_list["b_r_" + key] = numpy.zeros(shape=(output_num,), dtype="float32")
                self._parameter_list["b_z_" + key] = numpy.zeros(shape=(output_num,), dtype="float32")
                self._parameter_list["b_" + key] = numpy.zeros(shape=(output_num,), dtype="float32")
                self._parameter_list["H_Mem_" + key] = numpy.zeros(shape=(output_num,), dtype="float32")
                self._noise_factor["W_r_" + key] = noise_factor
                self._noise_factor["W_z_" + key] = noise_factor
                self._noise_factor["W_" + key] = noise_factor
                self._noise_factor["b_r_" + key] = noise_factor
                self._noise_factor["b_z_" + key] = noise_factor
                self._noise_factor["b_" + key] = noise_factor
                self._noise_factor["H_Mem_" + key] = noise_factor
            elif(connect_type == "lstm"):
                self._hidden_memory.add(key)
                true_output_num = output_num // 2
                input_size += true_output_num
                self._parameter_list["W_f_" + key] = numpy.random.normal(size=(true_output_num, input_size),
                        loc=0.0, scale=scale/numpy.sqrt(input_size))
                self._parameter_list["W_c_" + key] = numpy.random.normal(size=(true_output_num, input_size),
                        loc=0.0, scale=scale/numpy.sqrt(input_size))
                self._parameter_list["W_i_" + key] = numpy.random.normal(size=(true_output_num, input_size),
                        loc=0.0, scale=scale/numpy.sqrt(input_size))
                self._parameter_list["W_o_" + key] = numpy.random.normal(size=(true_output_num, input_size),
                        loc=0.0, scale=scale/numpy.sqrt(input_size))
                self._parameter_list["b_f_" + key] = numpy.zeros(shape=(true_output_num,), dtype="float32")
                self._parameter_list["b_c_" + key] = numpy.zeros(shape=(true_output_num,), dtype="float32")
                self._parameter_list["b_i_" + key] = numpy.zeros(shape=(true_output_num,), dtype="float32")
                self._parameter_list["b_o_" + key] = numpy.zeros(shape=(true_output_num,), dtype="float32")
                self._parameter_list["H_Mem_" + key] = numpy.zeros(shape=(output_num,), dtype="float32")
                self._noise_factor["W_f_" + key] = noise_factor
                self._noise_factor["W_c_" + key] = noise_factor
                self._noise_factor["W_i_" + key] = noise_factor
                self._noise_factor["W_o_" + key] = noise_factor
                self._noise_factor["b_f_" + key] = noise_factor
                self._noise_factor["b_c_" + key] = noise_factor
                self._noise_factor["b_i_" + key] = noise_factor
                self._noise_factor["b_o_" + key] = noise_factor
                self._noise_factor["H_Mem_" + key] = noise_factor
                #output size is special for LSTM
                self._hidden_length[key] = true_output_num
            elif(connect_type == "const"):
                self._hidden_length[key] = output_num
            elif(connect_type == "embedding"):
                self._parameter_list["b_" + key] = numpy.zeros(shape=(output_num,), dtype="float32")
                self._noise_factor["b_" + key] = noise_factor
                self._hidden_length[key] = output_num
            elif(connect_type == "tensor_embedding"):
                self._parameter_list["b_" + key] = numpy.zeros(shape=output_num, dtype="float32")
                self._noise_factor["b_" + key] = noise_factor
                self._hidden_length[key] = output_num
            else:
                raise Exception("No such connection type: %s"%connect_type)

        self.set_act_func()
        self.reset_rec_temp_mem()
        add_meta_lr()

    def reset_rec_temp_mem(self):
        self._hidden_temp_memory = dict()
        for key in self._hidden_memory:
            self._hidden_temp_memory[key] = numpy.copy(self._parameter_list["H_Mem_" + key])

    def reset_plastic_init(self):
        for key in self._model_structure:
            connect_type, input_layers, output_num, act_type, scale, noise_factor, pl_rule = self._model_structure[key]
            if(pl_rule is not None):
                if((pl_rule["type"] == "SABCD" or pl_rule["type"] == "ABCD")):
                    if(connect_type == "fc"):
                        input_size = self._parameter_list["W_" + key].shape[1]
                        self._parameter_list["W_" + key] = numpy.random.normal(
                            size=self._parameter_list["W_" + key].shape, loc=0.0, scale=scale/numpy.sqrt(input_size))
                    elif(connect_type == "rnn"):
                        input_size = self._parameter_list["W_" + key].shape[1]
                        output_size = self._parameter_list["W_" + key].shape[0]
                        self._parameter_list["W_" + key][:,:output_size] = numpy.random.normal(
                            size=(output_size, output_size), loc=0.0, scale=scale/numpy.sqrt(input_size))

    def set_act_func(self):
        def softmax(x):
            w = numpy.exp(x)
            return w/numpy.sum(w)

        def tanh_grad(x):
            return 1.0 - x * x

        def sigmoid_grad(x):
            return x * (1.0 - x)

        def relu_grad(x):
            return (x > 0).astype("float32")

        self._act_func["relu"] = lambda x:numpy.maximum(x, 0)
        self._act_func["softmax"] = softmax
        self._act_func["tanh"] = lambda x: numpy.tanh(x)
        self._act_func["sigmoid"] = lambda x: 0.5 * (numpy.tanh(0.5 * x) + 1.0)
        self._act_func["step"] = lambda x:numpy.asarray(x>0, dtype="float32")
        self._act_func["none"] = lambda x:x
        
        self._act_func_grad["tanh"] = tanh_grad
        self._act_func_grad["sigmoid"] = sigmoid_grad
        self._act_func_grad["relu"] = relu_grad
        self._act_func_grad["none"] = lambda x:1.0
        # Softmax Here are regard to be always connected to "log" as activation function, we will not calculate softmax gradient here
        self._act_func_grad["softmax"] = lambda x:1.0

    def get(self):
        return deepcopy(self._parameter_list)

    def set_params(self, params):
        self._parameter_list = deepcopy(params)
        self.reset_rec_temp_mem()
        #self.reset_plastic_init()

    def __copy__(self):
        return TemporalNN(weights = self.get())

    def load(self, file_name):
        file_op = open(file_name, "rb")
        self._parameter_list = pickle.load(file_op)
        file_op.close()
        self.reset_rec_temp_mem()

    def save(self, file_name):
        file_op = open(file_name, "wb")
        pickle.dump(self._parameter_list, file_op)
        file_op.close()

    def __repr__(self):
        return self.get()

    def run_inference_single(self, obs, hebb=False):
        hidden_values = {"input" : obs}
        output_records = dict()
        input_records = dict()
        modulars = dict()
        for key in self._model_structure:
            connect_type, input_layers, output_num, act_type, scale, noise_factor, _ = self._model_structure[key]
            if(connect_type == "rnn" or connect_type == "grnn" or connect_type == "lstm"):
                hidden_values[key] = self._hidden_temp_memory[key]

        for key in self._model_structure:
            connect_type, input_layers, output_num, act_type, scale, noise_factor, pl_rule = self._model_structure[key]
            #calculate the input
            inputs = []
            if(input_layers is not None):
                for input_layer in input_layers:
                    inputs.append(hidden_values[input_layer])
                inputs = numpy.concatenate(inputs, axis = 0)

            #perform forward
            if(connect_type == "fc"):
                hidden_values[key] = self._act_func[act_type](
                        numpy.matmul(self._parameter_list["W_" + key], inputs)
                        + self._parameter_list["b_" + key]
                        )
                input_records[key] = inputs
                output_records[key] = hidden_values[key]
            elif(connect_type == "rnn"):
                hidden_values[key] = self._act_func["tanh"](
                        numpy.matmul(self._parameter_list["W_" + key], 
                            numpy.concatenate([self._hidden_temp_memory[key], inputs], axis=0))
                        + self._parameter_list["b_" + key]
                        )
                input_records[key] = 0.50 * self._hidden_temp_memory[key] + 0.50
                output_records[key] = 0.50 * hidden_values[key] + 0.50
                self._hidden_temp_memory[key] = numpy.copy(hidden_values[key])
            elif(connect_type == "gru"):
                r_t = self._act_func["sigmoid"](
                        numpy.matmul(self._parameter_list["W_r_" + key], 
                            numpy.concatenate([self._hidden_temp_memory[key], inputs], axis=0))
                        + self._parameter_list["b_r_" + key]
                        )
                z_t = self._act_func["sigmoid"](
                        numpy.matmul(self._parameter_list["W_z_" + key], 
                            numpy.concatenate([self._hidden_temp_memory[key], inputs], axis=0))
                        + self._parameter_list["b_z_" + key]
                        )
                gated_hidden = self._hidden_temp_memory[key] * r_t
                h_t = self._act_func["tanh"](
                        numpy.matmul(self._parameter_list["W_" + key], 
                            numpy.concatenate([gated_hidden, inputs], axis=0))
                        + self._parameter_list["b_" + key]
                        )
                hidden_values[key] = h_t * z_t + self._hidden_temp_memory[key] * (1 - z_t)
                self._hidden_temp_memory[key] = numpy.copy(hidden_values[key])
            elif(connect_type == "lstm"):
                size = self._hidden_temp_memory[key].shape[-1] // 2
                i_t = self._act_func["sigmoid"](
                        numpy.matmul(self._parameter_list["W_i_" + key], 
                            numpy.concatenate([self._hidden_temp_memory[key][:size], inputs], axis=0))
                        + self._parameter_list["b_i_" + key]
                        )
                f_t = self._act_func["sigmoid"](
                        numpy.matmul(self._parameter_list["W_f_" + key], 
                            numpy.concatenate([self._hidden_temp_memory[key][:size], inputs], axis=0))
                        + self._parameter_list["b_f_" + key]
                        )
                o_t = self._act_func["sigmoid"](
                        numpy.matmul(self._parameter_list["W_o_" + key], 
                            numpy.concatenate([self._hidden_temp_memory[key][:size], inputs], axis=0))
                        + self._parameter_list["b_o_" + key]
                        )
                c_sharp = self._act_func["tanh"](
                        numpy.matmul(self._parameter_list["W_c_" + key], 
                            numpy.concatenate([self._hidden_temp_memory[key][:size], inputs], axis=0))
                        + self._parameter_list["b_c_" + key]
                        )
                c_t = f_t * self._hidden_temp_memory[key][size:] + i_t * c_sharp
                hidden_values[key] = o_t * self._act_func["tanh"](c_t)
                self._hidden_temp_memory[key] = numpy.concatenate([hidden_values[key], c_t], axis=-1)
            elif(connect_type == "const"):
                hidden_values[key] = numpy.ones(shape=(self._hidden_length[key],), dtype="float32")
            elif(connect_type == "embedding"):
                hidden_values[key] = numpy.copy(self._parameter_list["b_"+ key])
            elif(connect_type == "tensor_embedding"):
                hidden_values[key] = numpy.copy(self._parameter_list["b_"+ key])
            else:
                raise Exception("Unsupported Connect Type:" + connect_type)
        
        if(hebb):
            for key in self._model_structure:
                connect_type, input_layers, output_num, act_type, scale, noise_factor, pl_rule = self._model_structure[key]
                if(pl_rule is not None):
                    input_size = input_records[key].shape[0]
                    if(pl_rule["type"] == "SABCD"):
                        self._parameter_list["W_" + key][:,:input_size] += Plasticity_SABCD(
                                input_records[key], output_records[key],
                                hidden_values[pl_rule["S"]],
                                hidden_values[pl_rule["A"]],
                                hidden_values[pl_rule["B"]],
                                hidden_values[pl_rule["C"]],
                                hidden_values[pl_rule["D"]]
                                )
                    elif(pl_rule["type"] == "ABCD"):
                        self._parameter_list["W_" + key][:,:input_size] += Plasticity_ABCD(
                                input_records[key], output_records[key],
                                hidden_values[pl_rule["A"]],
                                hidden_values[pl_rule["B"]],
                                hidden_values[pl_rule["C"]],
                                hidden_values[pl_rule["D"]]
                                )
                    elif(pl_rule["type"] == "ABCDE"):
                        self._parameter_list["W_" + key][:,:input_size] += Plasticity_ABCDE(
                                input_records[key], output_records[key],
                                hidden_values[pl_rule["A"]],
                                hidden_values[pl_rule["B"]],
                                hidden_values[pl_rule["C"]],
                                hidden_values[pl_rule["D"]],
                                hidden_values[pl_rule["E"]]
                                )
                    elif(pl_rule["type"] == "SABCDE"):
                        self._parameter_list["W_" + key][:,:input_size] += Plasticity_SABCDE(
                                input_records[key], output_records[key],
                                hidden_values[pl_rule["S"]],
                                hidden_values[pl_rule["A"]],
                                hidden_values[pl_rule["B"]],
                                hidden_values[pl_rule["C"]],
                                hidden_values[pl_rule["D"]],
                                hidden_values[pl_rule["E"]]
                                )
                    elif(pl_rule["type"] == "S"):
                        self._parameter_list["W_" + key][:,:input_size] += Plasticity_S(
                                input_records[key], output_records[key],
                                hidden_values[pl_rule["S"]],
                                self._parameter_list["Meta_A"], self._parameter_list["Meta_B"],
                                self._parameter_list["Meta_C"], self._parameter_list["Meta_D"]
                                )
                    else:
                        raise Exception("No such plasticity type: %s" % pl_rule["type"])

        #print("forward time: %f, hebb_time: %f" % (forward_time_stop - forward_time_start, hebb_time - forward_time_stop))
        return hidden_values["output"], hidden_values

    def run_inference_batch(self, obses):
        self.reset_rec_temp_mem()
        hidden_values = {"input" : obses}
        input_records = dict()
        batch_size = obses.shape[0]
        preact_outputs = None
        for key in self._model_structure:
            connect_type, input_layers, output_num, act_type, scale, noise_factor, _ = self._model_structure[key]
            #calculate the input
            inputs = []
            if(input_layers is not None):
                for input_layer in input_layers:
                    inputs.append(hidden_values[input_layer])
                inputs = numpy.concatenate(inputs, axis = -1)
            input_records[key] = inputs

            if(connect_type == "fc"):
                hidden_values[key] = self._act_func[act_type](
                        numpy.matmul(input_records[key], numpy.transpose(self._parameter_list["W_" + key]))
                        + self._parameter_list["b_" + key]
                        )
            elif(connect_type == "const"):
                hidden_values[key] = numpy.ones(shape=(batch_size, self._hidden_length[key]), dtype="float32")
            elif(connect_type == "embedding"):
                hidden_values[key] = numpy.tile(self._parameter_list["b_" + key], (batch_size, 1))
            else:
                raise Exception("Unsupported Connect Type:" + connect_type)
        return hidden_values["output"], input_records, hidden_values

    def sync_rec_mem(self):
        for key in self._hidden_temp_memory:
            self._parameter_list["H_Mem_" + key] = numpy.copy(self._hidden_temp_memory[key])

    # Back propagation (Policy Gradient)
    def run_pg(self, alpha, inputs, actions, advantage):
        outputs, input_records, hiddens = self.run_inference_batch(inputs)
        batch_size = inputs.shape[0]
        grad = alpha * numpy.expand_dims(advantage, axis=1) * (outputs - actions)
        grads = {"output": grad}
        for key in reversed(list(self._model_structure.keys())):
            connect_type, input_layers, output_num, act_type, scale, noise_factor, _ = self._model_structure[key]
            if(connect_type == "fc" and act_type in self._act_func_grad):
                g_y = grads[key] * self._act_func_grad[act_type](hiddens[key])
                g_x = numpy.matmul(g_y, self._parameter_list["W_" + key])
                grad_w = (1.0 / batch_size) * numpy.matmul(numpy.transpose(g_y), input_records[key])
                grad_b = numpy.mean(g_y, axis=0)
                self._parameter_list["W_" + key] -= grad_w
                self._parameter_list["b_" + key] -= grad_b
                beg = 0
                for input_layer in input_layers:
                    end = beg + self._hidden_length[input_layer]
                    grads[input_layer] = g_x[:, beg:end]
                    beg = end
            else:
                raise Exception("Gradient Pipe Broken, Can only support FC with relu, sigmoid and tanh currently")
        #n_outputs, input_records, hiddens = self.run_inference_batch(inputs)
        #raise Exception("actions", actions, "\nadvantage", advantage, "\nold_outputs", outputs, "\noutputs", n_outputs)
        return

    # Back propagation (Policy Gradient)
    def run_discrete_pg(self, alpha, inputs, actions, advantage):
        outputs, input_records, hiddens = self.run_inference_batch(inputs)
        batch_size = inputs.shape[0]
        idxes = numpy.vstack([numpy.arange(actions.shape[0]), actions])
        sigma = numpy.zeros_like(outputs)
        sigma[idxes.T] = 1.0
        grad = alpha * numpy.expand_dims(advantage, axis=1) * (outputs - sigma)
        grads = {"output": grad}
        for key in reversed(list(self._model_structure.keys())):
            connect_type, input_layers, output_num, act_type, scale, noise_factor, _ = self._model_structure[key]
            if(connect_type == "fc" and act_type in self._act_func_grad):
                g_y = grads[key] * self._act_func_grad[act_type](hiddens[key])
                g_x = numpy.matmul(g_y, self._parameter_list["W_" + key])
                grad_w = (1.0 / batch_size) * numpy.matmul(numpy.transpose(g_y), input_records[key])
                grad_b = numpy.mean(g_y, axis=0)
                self._parameter_list["W_" + key] -= grad_w
                self._parameter_list["b_" + key] -= grad_b
                beg = 0
                for input_layer in input_layers:
                    end = beg + self._hidden_length[input_layer]
                    grads[input_layer] = g_x[:, beg:end]
                    beg = end
            else:
                raise Exception("Gradient Pipe Broken, Can only support FC with relu, sigmoid, softmax and tanh currently")
        return

    # Back propagation (Supservised Learning)
    def run_bp(self, alpha, inputs, labels):
        outputs, input_records, hiddens = self.run_inference_batch(inputs)
        batch_size = inputs.shape[0]
        grad = 2.0 * alpha * numpy.clip(outputs - labels, -1.0, 1.0)
        grads = {"output": grad}
        for key in reversed(list(self._model_structure.keys())):
            connect_type, input_layers, output_num, act_type, scale, noise_factor, _ = self._model_structure[key]
            if(connect_type == "fc" and act_type in self._act_func_grad):
                g_y = grads[key] * self._act_func_grad[act_type](hiddens[key])
                g_x = numpy.matmul(g_y, self._parameter_list["W_" + key])
                grad_w = (1.0 / batch_size) * numpy.matmul(numpy.transpose(g_y), input_records[key])
                grad_b = numpy.mean(g_y, axis=0)
                self._parameter_list["W_" + key] -= grad_w
                self._parameter_list["b_" + key] -= grad_b
                beg = 0
                for input_layer in input_layers:
                    end = beg + self._hidden_length[input_layer]
                    grads[input_layer] = g_x[:, beg:end]
                    beg = end
            else:
                raise Exception("Gradient Pipe Broken, Can only support FC with relu, sigmoid, softmax and tanh currently")
        return

    def add_evolution_noise(self, noise, learning_rate):
        for key in noise:
            self._parameter_list[key] += learning_rate * noise[key]
        return
