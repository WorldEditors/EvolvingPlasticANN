#!/usr/bin/env python
# coding=utf8
# File: config.py
import os
import math
from env_robot_navigating import WRNav

root = "./results/"
directory = root + "/workspace_WRNav/"
ent_factor = 1.0e-6
horizons = 100
noise = 0.01
input_neurons = 3

# Model Structure for EPRNN
model_structures = {
        "FC_1": ("fc", ["input"], 64, "relu", 1.0, 1.0e-2, None), 
        "FC_2": ("fc", ["FC_1"], 64, "relu", 1.0, 1.0e-2, None), 
        "Heb_A": ("tensor_embedding", None, (64, 64), "none", 1.0, 1.0e-3, None),
        "Heb_B": ("tensor_embedding", None, (64, 64), "none", 1.0, 1.0e-3, None),
        "Heb_C": ("tensor_embedding", None, (64, 64), "none", 1.0, 1.0e-3, None),
        "Heb_D": ("tensor_embedding", None, (64, 64), "none", 1.0, 1.0e-3, None),
        "RNN_Mod": ("fc", ["FC_2", "RNN_1"], 64, "sigmoid", 1.0, 1.0e-2, None), 
        "RNN_1": ("rnn", ["FC_2"], 64, "tanh", 1.0, 1.0e-2, 
            {"type": "SABCD",
                "S": "RNN_Mod",
                "A": "Heb_A",
                "B": "Heb_B",
                "C": "Heb_C",
                "D": "Heb_D",
                }),
        "FC_3": ("fc", ["RNN_1"], 64, "sigmoid", 1.0, 1.0e-2, None),
        "output": ("fc", ["FC_3"], 2, "none", 1.0, 1.0e-2, None)
        }

# Refer to config_SeqPred_task.py for other configurations

#If no load_model is specified, the model is random initialized
#load_model = model.dat

#Address for xparl servers, do "xparl start " in your server
server = "localhost:8010"

#True Batch size = Actor_number * batch_size
actor_number = 400
batch_size = 1
task_sub_iterations = 1
inner_rollouts = [(0.0, "TRAIN", True), (1.0, "TEST", False)]

#The task pattern are kept still for that much steps
pattern_renew = 4
pattern_retain_iterations = 1

#Select the inner-loop type, for PRNN / RNN / LSTM / EPMLP select "recursive", for ES-MAML select "pg"
adapt_type = "recursive"
#adapt_type = "pg"

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
    return list(action) + [info["observation"]]

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
