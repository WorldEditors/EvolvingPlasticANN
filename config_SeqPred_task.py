#!/usr/bin/env python
# coding=utf8
# File: config.py
import os
import math
from env_sequence_predicting import SequencePredicting

root = "./results/"
directory = root + "/workspace_SeqPred/"
ent_factor = 1.0e-6
K = 10
N = 20
l = 3
input_neurons = l

# Model Structure for EPRNN
model_structures = {
        "FC_1": ("fc", ["input"], 64, "relu", 1.0, 1.0e-2, None), 
        "Heb_A": ("tensor_embedding", None, (64, 64), "none", 1.0, 1.0e-3, None),
        "Heb_B": ("tensor_embedding", None, (64, 64), "none", 1.0, 1.0e-3, None),
        "Heb_C": ("tensor_embedding", None, (64, 64), "none", 1.0, 1.0e-3, None),
        "Heb_D": ("tensor_embedding", None, (64, 64), "none", 1.0, 1.0e-3, None),
        "RNN_Mod": ("fc", ["FC_1", "RNN_1"], 64, "sigmoid", 1.0, 1.0e-2, None), 
        "RNN_1": ("rnn", ["FC_1"], 64, "tanh", 1.0, 1.0e-2, 
            {"type": "SABCD",
                "S": "RNN_Mod",
                "A": "Heb_A",
                "B": "Heb_B",
                "C": "Heb_C",
                "D": "Heb_D",
                }),
        "FC_2": ("fc", ["RNN_1"], 64, "sigmoid", 1.0, 1.0e-2, None),
        "output": ("fc", ["FC_2"], l, "none", 1.0, 1.0e-2, None)
        }

# Model Structure for EPMLP
#model_structures = {
#        "FC_1": ("fc", ["input"], 64, "relu", 1.0, 1.0e-2, None), 
#        "Heb_A": ("tensor_embedding", None, (64, 64), "none", 1.0, 1.0e-3, None),
#        "Heb_B": ("tensor_embedding", None, (64, 64), "none", 1.0, 1.0e-3, None),
#        "Heb_C": ("tensor_embedding", None, (64, 64), "none", 1.0, 1.0e-3, None),
#        "Heb_D": ("tensor_embedding", None, (64, 64), "none", 1.0, 1.0e-3, None),
#        "FC_Mod": ("fc", ["FC_1", "RNN_1"], 64, "sigmoid", 1.0, 1.0e-2, None), 
#        "FC_2": ("fc", ["FC_1"], 64, "tanh", 1.0, 1.0e-2, 
#            {"type": "SABCD",
#                "S": "RNN_Mod",
#                "A": "Heb_A",
#                "B": "Heb_B",
#                "C": "Heb_C",
#                "D": "Heb_D",
#                }),
#        "FC_3": ("fc", ["FC_2"], 64, "sigmoid", 1.0, 1.0e-2, None),
#        "output": ("fc", ["FC_3"], l, "none", 1.0, 1.0e-2, None)
#        }

# Model Structure for ES-RNN & ES-LSTM & ES-MAML
#model_structures = {
#        "FC_1": ("fc", ["input"], 64, "relu", 1.0, 1.0e-2, None), 
#        "Versatile": ("rnn", ["FC_1"], 64, "tanh", 1.0, 1.0e-2, None),
#        #"Versatile": ("lstm", ["FC_1"], 64, "tanh", 1.0, 1.0e-2, None),
#        #"Versatile": ("fc", ["FC_1"], 64, "tanh", 1.0, 1.0e-2, None),
#        "FC_2": ("fc", ["Versatile"], 64, "sigmoid", 1.0, 1.0e-2, None),
#        "output": ("fc", ["FC_2"], l, "none", 1.0, 1.0e-2, None)
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

#Select the inner-loop type, for PRNN/EPMLP/ES-RNN/ES-LSTM select "recursive", for ES-MAML select "bp"
adapt_type = "recursive"
#adapt_type = "bp"

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
        return list(info["observation"])
    else:
        return action

#Sampled Tasks for meta-training
def gen_pattern(pattern_number=16):
    return [SequencePredicting.gen_pattern(l) for _ in range(pattern_number)]

#Sampled Tasks for meta-testing
def test_patterns(pattern_number=1600):
    return [SequencePredicting.gen_pattern(l) for _ in range(pattern_number)]

def game():
    return SequencePredicting(l, K, N)
