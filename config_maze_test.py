#!/usr/bin/env python
# coding=utf8
# File: config.py
import os
import math
import random
import numpy
from envs.env_maze import  MazeTask
from envs.env_maze import gen_pattern as gen_single
from envs.env_maze import output_to_action
from envs.env_maze import obs_to_input
from epann.utils import categorical
from models_zoo import ModelRNNBase1, ModelLSTMBase1, ModelFCBase2
from models_zoo import ModelPRNNPreMod, ModelPRNNAfterMod, ModelPRNNNoMod
from models_zoo import ModelPFCPreMod, ModelPFCAfterMod, ModelPFCNoMod
from gen_train_test_patterns import gen_patterns

root = "./"
ent_factor = 1.0e-6

# Model Structure for EPRNN
# Hebbian-Type: 1. alpha ABCD; 2. Eligibility Traces; 3. Decomposed Plasticity; 4. Evolving & Merging
#model = ModelLSTMBase1(input_shape=(15,), output_shape=(5,), hidden_size=64, extra_hidden_size=64, output_activation="none", init_scale=0.05)
model = ModelRNNBase1(input_shape=(15,), output_shape=(5,), hidden_size=64, output_activation="none", initialize_settings='R', init_scale=0.0)
#model = ModelPRNNAfterMod(input_shape=(15,), output_shape=(5,), hidden_size=64, output_activation="none", initialize_settings='R', hebbian_type=3, init_scale=0.05)
#model = ModelPRNNNoMod(input_shape=(15,), output_shape=(5,), hidden_size=64, output_activation="none", initialize_settings='R', hebbian_type=3, init_scale=0.05)
# Hebbian-Type: 1. alpha ABCD; 2. Eligibility Traces; 3. Decomposed Plasticity; 4. Evolving & Merging
#model = ModelPRNNAfterMod(input_shape=(15,), output_shape=(5,), hidden_size=64, output_activation="none", initialize_settings='R', hebbian_type=3, init_scale=0.05)

# Refer to config_SeqPred_task.py for other configurations

#If no load_model is specified, the model is random initialized
test_load_model = "./models/maze21_l_rnn.dat"
#test_load_model = "./results/workspace_maze_l_decprnn_cdn/models/model.000500.dat"

# For Model Conversion Only
model_from = ModelPRNNAfterMod(input_shape=(15,), output_shape=(5,), hidden_size=64, output_activation="none", initialize_settings='R', hebbian_type=1, init_scale=0.05)
model_to = ModelPRNNAfterMod(input_shape=(15,), output_shape=(5,), hidden_size=64, output_activation="none", initialize_settings='R', hebbian_type=4, init_scale=0.05)
load_model_from = ""
save_model_to = ""

# Refer to config_SeqPred_task.py for other configurations

#Address for xparl servers, do "xparl start " in your server
#server = "localhost:8010"
server = "10.216.186.16:8010"

#True Batch size = Actor_number * batch_size
actor_number = 80
batch_size = 1
inner_rollouts = [(0.0, "TRAIN", True), (0.0, "TRAIN", True), (0.16, "TEST", True),
        (0.22, "TEST", True), (0.36, "TEST", True), (0.64, "TEST", True),
        (0.8, "TEST", True), (1.0, "TEST", True)
        ]

#The task pattern are kept still for that much steps
pattern_renew = 16
pattern_retain_iterations = 1

#Select the inner-loop type, for PRNN / RNN / LSTM / EPMLP select "forward", for ES-MAML select "policy_gradient_continuous"
adapt_type = "forward"

evolution_pool_size = 400
evolution_topk_size = 200
# CMA-ES initial noise variance
evolution_step_size = 0.01
# CMA-ES hyper-parameter
evolution_lr = 0.02

# Model Saving Intervals
save_iter = 500
# Maximum Iterations
max_iter = 15000
# Intervals for meta-testing
test_iter = 100

#Sampled Tasks for meta-testing
def test_patterns():
    return gen_patterns(n=2048, file_name="./demo/tasks/2048_maze21.dat")

def game():
    return MazeTask()
