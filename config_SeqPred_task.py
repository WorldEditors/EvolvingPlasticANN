#!/usr/bin/env python
# coding=utf8
# File: config.py
import os
import math
from env_sequence_predicting import SequencePredicting
from layers import *

root = "./results/"
directory = root + "/workspace_SeqPred/"
ent_factor = 1.0e-6
K = 10
N = 20
l = 3

# Model Structure for EPRNN
model_structures = {
        "observation": Input(l),
        "fc_1": FC(64, param_name="fc_1", input_keys="observation", act_type="relu"),
        "heb_a": TensorEmb((64,64), param_name="heb_a", evolution_noise_scale=1.0e-3),
        "heb_b": TensorEmb((64,64), param_name="heb_b", evolution_noise_scale=1.0e-3),
        "heb_c": TensorEmb((64,64), param_name="heb_c", evolution_noise_scale=1.0e-3),
        "heb_d": TensorEmb((64,64), param_name="heb_d", evolution_noise_scale=1.0e-3),
        "rnn_1": Mem(64, param_name="rnn_1"),
        "rnn_input": Concat(128, input_keys=["fc_1", "rnn_1"]),
        "rnn_mod": FC(64, param_name="rnn_mod", input_keys="rnn_input", act_type="sigmoid"),
        "rnn": FC(64, param_name="rnn", input_keys="rnn_input", act_type="tanh",output_keys=["rnn_1"],
            pl_dict= {"type": "SABCD",
                "input_start": 64,
                "input_end" : 128,
                "S": "rnn_mod",
                "A": "heb_a",
                "B": "heb_b",
                "C": "heb_c",
                "D": "heb_d"}
            ),
        "fc_2": FC(64, param_name="fc_2", input_keys="rnn", act_type="relu"),
        "output": FC(l, param_name="fc_3", input_keys="fc_2", act_type="none")
        }

# Model Structure for EPMLP
#model_structures = {
#        "observation": Input(l),
#        "fc_1": FC(64, param_name="fc_1", input_keys="observation", act_type="relu"),
#        "heb_a": TensorEmb((64,64), param_name="heb_a", evolution_noise_scale=1.0e-3),
#        "heb_b": TensorEmb((64,64), param_name="heb_b", evolution_noise_scale=1.0e-3),
#        "heb_c": TensorEmb((64,64), param_name="heb_c", evolution_noise_scale=1.0e-3),
#        "heb_d": TensorEmb((64,64), param_name="heb_d", evolution_noise_scale=1.0e-3),
#        "heb_mod": FC(64, param_name="heb_mod", input_keys="fc_1", act_type="sigmoid"),
#        "fc_heb": FC(64, param_name="fc_heb", input_keys="fc_1", act_type="tanh",
#            pl_dict= {"type": "SABCD",
#                "S": "heb_mod",
#                "A": "heb_a",
#                "B": "heb_b",
#                "C": "heb_c",
#                "D": "heb_d"}
#            ),
#        "fc_2": FC(64, param_name="fc_2", input_keys="fc_heb", act_type="relu"),
#        "output": FC(l, param_name="fc_3", input_keys="fc_2", act_type="none")
#        }

# Model Structure for ES-RNN 
#model_structures = {
#        "observation": Input(l),
#        "fc_1": FC(64, param_name="fc_1", input_keys="observation", act_type="relu"),
#        "rnn_1": Mem(64, param_name="rnn_1"),
#        "rnn_input": Concat(128, input_keys=["fc_1", "rnn_1"]),
#        "rnn": FC(64, param_name="rnn", input_keys="rnn_input", act_type="tanh",output_keys=["rnn_1"]),
#        "fc_2": FC(64, param_name="fc_2", input_keys="rnn", act_type="relu"),
#        "output": FC(l, param_name="fc_3", input_keys="fc_2", act_type="none")
#        }

# Model Structure for ES-LSTM 
#model_structures = {
#        "observation": Input(l),
#        "fc_1": FC(64, param_name="fc_1", input_keys="observation", act_type="relu"),
#        "lstm_h": Mem(64, param_name="lstm_h"),
#        "lstm_c": Mem(64, param_name="lstm_c"),
#        "lstm_i": Concat(128, input_keys=["fc_1", "lstm_h"]),
#        "lstm_g_i": FC(64, param_name="lstm_g_i", input_keys="lstm_i", act_type="sigmoid"),
#        "lstm_g_f": FC(64, param_name="lstm_g_f", input_keys="lstm_i", act_type="sigmoid"),
#        "lstm_g_o": FC(64, param_name="lstm_g_o", input_keys="lstm_i", act_type="sigmoid"),
#        "lstm_c_sharp": FC(64, param_name="lstm_c_sharp", input_keys="lstm_i", act_type="tanh"),
#        "lstm_c_1": EleMul(64, input_keys=["lstm_c_sharp", "lstm_g_i"]),
#        "lstm_c_2": EleMul(64, input_keys=["lstm_h", "lstm_g_f"]),
#        "lstm_c_out": SumPooling(64, input_keys=["lstm_c_1", "lstm_c_2"], output_keys="lstm_c"),
#        "lstm_c_act": ActLayer(64, input_keys="lstm_c", act_type="tanh"),
#        "lstm_out": EleMul(64, input_keys=["lstm_c_act", "lstm_g_o"]),
#        "fc_2": FC(64, param_name="fc_2", input_keys="lstm_out", act_type="relu"),
#        "output": FC(l, param_name="fc_3", input_keys="fc_2", act_type="none")
#        }

# Model Structure for ES-MAML
#model_structures = {
#        "observation": Input(l),
#        "fc_1": FC(64, param_name="fc_1", input_keys="observation", act_type="relu"),
#        "fc_2": FC(64, param_name="fc_2", input_keys="fc_1", act_type="relu"),
#        "fc_3": FC(64, param_name="fc_3", input_keys="fc_2", act_type="relu"),
#        "output": FC(l, param_name="fc_4", input_keys="fc_3", act_type="none"),
#        "inner_learning_rate": {"initial_parameter": [0.1, 0.1, 0.1, 0.1], "noise": 1.0e-2}
#        }

#If no load_model is specified, the model is random initialized
#load_model = model.dat

#Address for xparl servers, do "xparl start " in your server
server = "localhost:8010"

#True Batch size = Actor_number * batch_size
actor_number = 400
batch_size = 1
task_sub_iterations = 1
inner_rollouts = [(0.0, "TRAIN", True), (1.0, "TEST", False)]
# Doing Hebbian learning in both meta-training-train and meta-training-test
#inner_rollouts = [(0.0, "TRAIN", True), (1.0, "TEST", True)]

#The task pattern are kept still for that much steps
pattern_renew = 16
pattern_retain_iterations = 1

#Select the inner-loop type, for PRNN/EPMLP/ES-RNN/ES-LSTM select "forward", for ES-MAML select "supervised_learning"
adapt_type = "forward"

evolution_pool_size = 400
learning_rate = 0.02

save_iter = 100
max_iter = 15000
#Intervals for meta-testing
test_iter = 100

#Transform from raw output to actions, use is_tra if it is different in meta-training-train and meta-training-test
def output_to_action(output_list, is_tra):
    return list(output_list)

#Transform the observation, previous action, and info into observation, pattern is not allowed to use in meta-learning
def obs_to_input(obs, action, info, pattern):
    if(info["is_observation_valid"]):
        return list(info["observation"]), list(info["label"])
    else:
        return action, None

#Sampled Tasks for meta-training
def gen_pattern(pattern_number=16):
    return [SequencePredicting.gen_pattern(l) for _ in range(pattern_number)]

#Sampled Tasks for meta-testing
def test_patterns(pattern_number=1600):
    return [SequencePredicting.gen_pattern(l) for _ in range(pattern_number)]

def game():
    return SequencePredicting(l, K, N)
