#!/usr/bin/env python
# coding=utf8
# File: config.py
import os
import math
from env_robot_navigating import WRNav
from layers import *

root = "./results/"
directory = root + "/workspace_WRNav/"
ent_factor = 1.0e-6
horizons = 100
noise = 0.01

# Model Structure for EPRNN
model_structures = {
        "observation": Input(3),
        "fc_1": FC(64, param_name="fc_1", input_keys="observation", act_type="relu"),
        "fc_2": FC(64, param_name="fc_2", input_keys="fc_1", act_type="relu"),
        "heb_a": TensorEmb((64,64), param_name="heb_a", evolution_noise_scale=1.0e-3),
        "heb_b": TensorEmb((64,64), param_name="heb_b", evolution_noise_scale=1.0e-3),
        "heb_c": TensorEmb((64,64), param_name="heb_c", evolution_noise_scale=1.0e-3),
        "heb_d": TensorEmb((64,64), param_name="heb_d", evolution_noise_scale=1.0e-3),
        "rnn_1": Mem(64, param_name="rnn_1"),
        "rnn_input": Concat(128, input_keys=["fc_2", "rnn_1"]),
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
        "fc_3": FC(64, param_name="fc_3", input_keys="rnn", act_type="relu"),
        "output": FC(2, param_name="fc_4", input_keys="fc_3", act_type="tanh")
        }

# Model Structure for ES-MAML
#model_structures = {
#        "observation": Input(3),
#        "fc_1": FC(64, param_name="fc_1", input_keys="observation", act_type="relu"),
#        "fc_2": FC(64, param_name="fc_2", input_keys="fc_1", act_type="relu"),
#        "fc_3": FC(64, param_name="fc_3", input_keys="fc_2", act_type="relu"),
#        "fc_4": FC(64, param_name="fc_4", input_keys="fc_3", act_type="relu"),
#        "output": FC(2, param_name="fc_5", input_keys="fc_4", act_type="none"),
#        "inner_learning_rate": {"initial_parameter": [0.1, 0.1, 0.1, 0.1], "noise": 1.0e-2}
#        }

# Refer to config_SeqPred_task.py for other configurations

#If no load_model is specified, the model is random initialized
#load_model = model.dat

#Address for xparl servers, do "xparl start " in your server
#server = "localhost:8010"
server = "10.216.186.16:8010"

#True Batch size = Actor_number * batch_size
actor_number = 400
batch_size = 1
task_sub_iterations = 1
inner_rollouts = [(0.0, "TRAIN", True), (1.0, "TEST", False)]

#The task pattern are kept still for that much steps
pattern_renew = 4
pattern_retain_iterations = 1

#Select the inner-loop type, for PRNN / RNN / LSTM / EPMLP select "forward", for ES-MAML select "policy_gradient_continuous"
adapt_type = "forward"
#adapt_type = "policy_gradient_continuous"

evolution_pool_size = 400
learning_rate = 0.02

save_iter = 100
max_iter = 15000
#Intervals for meta-testing
test_iter = 100

#Transform from raw output to actions, use is_tra if it is different in meta-training-train and meta-training-test
def output_to_action(output_list, is_tra):
    return output_list.tolist()

#Transform the observation, previous action, and info into observation, pattern is not allowed to use in meta-learning
def obs_to_input(obs, action, info, pattern):
    return list(action) + [info["observation"]], None

#Sampled Tasks for meta-training
def gen_pattern(pattern_number=4):
    return [WRNav.gen_pattern() for _ in range(pattern_number)]

#Sampled Tasks for meta-testing
def test_patterns(pattern_number=1600):
    return [WRNav.gen_pattern() for _ in range(pattern_number)]

def game():
    return WRNav(horizons, noise)

# This reward normalization is specially for ES-MAML
GAMMA_ARRAY_ST = numpy.logspace(0, 9, 10, base=0.90, endpoint=0)
def rewards2adv(reward_list):
    length = len(reward_list)
    add_len = GAMMA_ARRAY_ST.shape[0]
    reward_list.extend([reward_list[-1] for _ in range(add_len)])
    adv = [numpy.sum(numpy.asarray(reward_list[i:i+add_len]) * GAMMA_ARRAY_ST) for i in range(length)]
    return list(adv)
