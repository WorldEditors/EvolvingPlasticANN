import numpy
import math
from models import Models
from layers import Layers, Memory, FC, Pooling, Hebbian, PlasticFC, PlasticUnFC, PlasticRNN, LSTM, SimpleRNN, Conv, VSML 
from activation import ActFunc

class ModelPRNNAfterMod(Models):
    def build_model(self):
        self.l1 = self.add_layer(PlasticRNN, param_name_prefix="PRNN_1", output_shape=(self.hidden_size,), input_shape=self.input_shape, activation="tanh", 
                initialize_settings=self.initialize_settings, hebbian_type=self.hebbian_type, 
                initialize_hyper_parameter=1.0,
                param_init_scale=self.init_scale)
        self.l2 = self.add_layer(FC, param_name_prefix="FC_1", output_shape=self.output_shape, input_shape=(self.hidden_size,), activation=self.output_activation,
                param_init_scale=self.init_scale)
        self.l3 = self.add_layer(FC, param_name_prefix="FC_M", output_shape=(2,), input_shape=(self.hidden_size,), activation="sigmoid",
                param_init_scale=self.init_scale)
    
    def forward(self, inputs):
        outputs = self.l1(numpy.array(inputs))
        mod = self.l3(outputs)
        outputs = self.l2(outputs)
        #print(mod)
        self.l1.learn(modulator={"h":mod[0], "x":mod[1]})
        return outputs

class ModelPRNNPreMod(Models):
    def build_model(self):
        scale = 0.01
        self.l1 = self.add_layer(PlasticRNN, param_name_prefix="PRNN_1", output_shape=(self.hidden_size,), input_shape=self.input_shape, activation="tanh", 
                initialize_settings=self.initialize_settings, hebbian_type=self.hebbian_type, 
                initialize_hyper_parameter=1.0,
                param_init_scale=self.init_scale)
        self.l2 = self.add_layer(FC, param_name_prefix="FC_1", output_shape=self.output_shape, input_shape=(self.hidden_size,), activation=self.output_activation,
                param_init_scale=self.init_scale)
        self.lmh = self.add_layer(FC, param_name_prefix="FC_Mh", output_shape=(1,), input_shape=self.input_shape, activation="sigmoid",
                param_init_scale=self.init_scale)
        self.lmx = self.add_layer(FC, param_name_prefix="FC_Mx", output_shape=(1,), input_shape=(self.hidden_size,), activation="sigmoid",
                param_init_scale=self.init_scale)
    
    def forward(self, inputs):
        modx = self.lmh(numpy.array(inputs))
        modh = self.lmx(self.l1.mem())
        outputs = self.l1(numpy.array(inputs))
        outputs = self.l2(outputs)
        self.l1.learn(modulator={"h":modh, "x":modx})
        return outputs

class ModelPRNNNoMod(Models):
    def build_model(self):
        self.l1 = self.add_layer(PlasticRNN, param_name_prefix="PRNN_1", output_shape=(self.hidden_size,), input_shape=self.input_shape, activation="tanh", 
                initialize_settings=self.initialize_settings, hebbian_type=self.hebbian_type, 
                initialize_hyper_parameter=1.0,
                param_init_scale=self.init_scale)
        self.l2 = self.add_layer(FC, param_name_prefix="FC_1", output_shape=self.output_shape, input_shape=(self.hidden_size,), activation=self.output_activation,
                param_init_scale=self.init_scale)
    
    def forward(self, inputs):
        outputs = self.l1(numpy.array(inputs))
        outputs = self.l2(outputs)
        self.l1.learn()
        return outputs

class ModelPFCPreMod(Models):
    def build_model(self):
        self.l1 = self.add_layer(PlasticFC, param_name_prefix="PFC_1", output_shape=(self.hidden_size,), input_shape=self.input_shape, activation="tanh",
                initialize_settings=self.initialize_settings, hebbian_type=self.hebbian_type, 
                initialize_hyper_parameter=1.0,
                param_init_scale=self.init_scale)
        self.l2 = self.add_layer(PlasticFC, param_name_prefix="PFC_2", output_shape=(self.hidden_size,), input_shape=(self.hidden_size,), activation="tanh",
                initialize_settings=self.initialize_settings, hebbian_type=self.hebbian_type, 
                initialize_hyper_parameter=1.0,
                param_init_scale=self.init_scale)
        self.l3 = self.add_layer(FC, param_name_prefix="FC_1", output_shape=self.output_shape, input_shape=(self.hidden_size,), activation=self.output_activation,
                param_init_scale=self.init_scale)
        self.lm1 = self.add_layer(FC, param_name_prefix="FC_Mh", output_shape=(1,), input_shape=self.input_shape, activation="sigmoid",
                param_init_scale=self.init_scale)
        self.lm2 = self.add_layer(FC, param_name_prefix="FC_Mx", output_shape=(1,), input_shape=(self.hidden_size,), activation="sigmoid",
                param_init_scale=self.init_scale)
    
    def forward(self, inputs):
        mod1 = self.lm1(numpy.array(inputs))
        outputs = self.l1(numpy.array(inputs))
        mod2 = self.lm2(outputs)
        outputs = self.l2(outputs)
        outputs = self.l3(outputs)
        self.l1.learn(modulator=mod1)
        self.l2.learn(modulator=mod2)
        return outputs

class ModelPFCAfterMod(Models):
    def build_model(self):
        self.l1 = self.add_layer(PlasticFC, param_name_prefix="PFC_1", output_shape=(self.hidden_size,), input_shape=self.input_shape, activation="tanh",
                initialize_settings=self.initialize_settings, hebbian_type=self.hebbian_type, 
                initialize_hyper_parameter=1.0,
                param_init_scale=self.init_scale)
        self.l2 = self.add_layer(PlasticFC, param_name_prefix="PFC_2", output_shape=(self.hidden_size,), input_shape=(self.hidden_size,), activation="tanh",
                initialize_settings=self.initialize_settings, hebbian_type=self.hebbian_type, 
                initialize_hyper_parameter=1.0,
                param_init_scale=self.init_scale)
        self.l3 = self.add_layer(FC, param_name_prefix="FC_1", output_shape=self.output_shape, input_shape=(self.hidden_size,), activation=self.output_activation,
                param_init_scale=self.init_scale)
        self.lm1 = self.add_layer(FC, param_name_prefix="FC_Mh", output_shape=(1,), input_shape=(self.hidden_size,), activation="sigmoid",
                param_init_scale=self.init_scale)
        self.lm2 = self.add_layer(FC, param_name_prefix="FC_Mx", output_shape=(1,), input_shape=(self.hidden_size,), activation="sigmoid",
                param_init_scale=self.init_scale)
    
    def forward(self, inputs):
        outputs = self.l1(numpy.array(inputs))
        mod1 = self.lm1(outputs)
        outputs = self.l2(outputs)
        mod2 = self.lm2(outputs)
        outputs = self.l3(outputs)
        self.l1.learn(modulator=mod1)
        self.l2.learn(modulator=mod2)
        return outputs

class ModelPFCNoMod(Models):
    def build_model(self):
        self.l1 = self.add_layer(PlasticFC, param_name_prefix="PFC_1", output_shape=(self.hidden_size,), input_shape=self.input_shape, activation="tanh",
                initialize_settings=self.initialize_settings, hebbian_type=self.hebbian_type, 
                initialize_hyper_parameter=1.0,
                param_init_scale=self.init_scale)
        self.l2 = self.add_layer(PlasticFC, param_name_prefix="PFC_2", output_shape=(self.hidden_size,), input_shape=(self.hidden_size,), activation="tanh",
                initialize_settings=self.initialize_settings, hebbian_type=self.hebbian_type, 
                initialize_hyper_parameter=1.0,
                param_init_scale=self.init_scale)
        self.l3 = self.add_layer(FC, param_name_prefix="FC_1", output_shape=self.output_shape, input_shape=(self.hidden_size,), activation=self.output_activation,
                param_init_scale=self.init_scale)
    
    def forward(self, inputs):
        outputs = self.l1(numpy.array(inputs))
        outputs = self.l2(outputs)
        outputs = self.l3(outputs)
        self.l1.learn()
        self.l2.learn()
        return outputs

class ModelFCBase1(Models):
    def build_model(self):
        scale = 0.01
        self.l1 = self.add_layer(FC, param_name_prefix="FC_1", output_shape=(self.hidden_size,), input_shape=self.input_shape, activation="tanh",
                param_init_scale=self.init_scale)
        self.l2 = self.add_layer(FC, param_name_prefix="FC_2", output_shape=self.output_shape, input_shape=(self.hidden_size,), activation=self.output_activation,
                param_init_scale=self.init_scale)
    
    def forward(self, inputs):
        outputs = self.l1(numpy.array(inputs))
        outputs = self.l2(outputs)
        return outputs

class ModelFCBase2(Models):
    def build_model(self):
        scale = 0.01
        self.l1 = self.add_layer(FC, param_name_prefix="FC_1", output_shape=(self.hidden_size,), input_shape=self.input_shape, activation="tanh",
                param_init_scale=self.init_scale)
        self.l2 = self.add_layer(FC, param_name_prefix="FC_2", output_shape=(self.hidden_size,), input_shape=(self.hidden_size,), activation="tanh",
                param_init_scale=self.init_scale)
        self.l3 = self.add_layer(FC, param_name_prefix="FC_3", output_shape=self.output_shape, input_shape=(self.hidden_size,), activation=self.output_activation,
                param_init_scale=self.init_scale)
    
    def forward(self, inputs):
        outputs = self.l1(numpy.array(inputs))
        outputs = self.l2(outputs)
        outputs = self.l3(outputs)
        return outputs

class ModelRNNBase1(Models):
    def build_model(self):
        self.l1 = self.add_layer(SimpleRNN, param_name_prefix="RNN_1", output_shape=(self.hidden_size,), input_shape=self.input_shape, activation="tanh",
                param_init_scale=self.init_scale)
        self.l2 = self.add_layer(FC, param_name_prefix="FC_2", output_shape=self.output_shape, input_shape=(self.hidden_size,), activation=self.output_activation,
                param_init_scale=self.init_scale)
    
    def forward(self, inputs):
        outputs = self.l1(inputs)
        outputs = self.l2(outputs)
        return outputs

class ModelLSTMBase1(Models):
    def build_model(self):
        self.l1 = self.add_layer(LSTM, param_name_prefix="LSTM_1", output_shape=(self.hidden_size,), input_shape=self.input_shape, activation="tanh",
                param_init_scale=self.init_scale)
        self.l2 = self.add_layer(FC, param_name_prefix="FC_2", output_shape=self.output_shape, input_shape=(self.hidden_size,), activation=self.output_activation,
                param_init_scale=self.init_scale)
    
    def forward(self, inputs):
        outputs = self.l1(inputs)
        outputs = self.l2(outputs)
        return outputs

class ModelLSTMBase2(Models):
    def build_model(self):
        scale = math.sqrt(6 / (self.extra_hidden_size + self.hidden_size))
        self.l1 = self.add_layer(FC, param_name_prefix="FC_1", output_shape=(self.extra_hidden_size,), input_shape=self.input_shape, activation="relu",
                param_init_scale=self.init_scale)
        self.l2 = self.add_layer(LSTM, param_name_prefix="LSTM_1", output_shape=(self.hidden_size,), input_shape=(self.extra_hidden_size,), activation="tanh",
                param_init_scale=self.init_scale)
        self.l3 = self.add_layer(FC, param_name_prefix="FC_2", output_shape=self.output_shape, input_shape=(self.hidden_size,), activation=self.output_activation,
                param_init_scale=self.init_scale)
    
    def forward(self, inputs):
        outputs = self.l1(inputs)
        outputs = self.l2(outputs)
        outputs = self.l3(outputs)
        return outputs

class ModelVSML(Models):
    def build_model(self):
        scale = 0.05
        self.l1 = self.add_layer(VSML, param_name_prefix="VSML", output_shape=(self.hidden_size,), input_shape=self.input_shape,
                param_init_scale=scale, inner_size = self.inner_size, m_size=self.m_size)
        self.l2 = self.add_layer(VSML, param_name_prefix="VSML", output_shape=self.output_shape, input_shape=(self.hidden_size,),
                param_init_scale=scale, inner_size = self.inner_size, m_size=self.m_size)
        self.l3 = ActFunc(self.output_activation)
    
    def forward(self, inputs):
        outputs = self.l1(inputs)
        outputs = self.l2(outputs)
        outputs = self.l3(outputs)
        return outputs
