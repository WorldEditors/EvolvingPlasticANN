#!/usr/bin/env python
# coding=utf8
# File: config.py
import os
import math
import random
import numpy
from envs.env_maze import  MazeTask
from envs.env_maze import output_to_action
from envs.env_maze import obs_to_input
from epann.utils import categorical
from models_zoo import ModelRNNBase1, ModelLSTMBase1, ModelFCBase2
from models_zoo import ModelPRNNPreMod, ModelPRNNAfterMod, ModelPRNNNoMod
from models_zoo import ModelPFCPreMod, ModelPFCAfterMod, ModelPFCNoMod
from gen_train_test_patterns import resample_maze9, resample_maze15, resample_maze21, import_mazes

root = "./results"
directory = root + "/workspace_maze/"

# Model Structure for EPRNN
#model = ModelFCBase2(input_shape=(15,), output_shape=(5,), hidden_size=64, output_activation="none", initialize_settings='R', init_scale=0.05)
#model = ModelLSTMBase1(input_shape=(15,), output_shape=(5,), hidden_size=64, extra_hidden_size=64, output_activation="none", init_scale=0.05)
#model = ModelRNNBase1(input_shape=(15,), output_shape=(5,), hidden_size=64, output_activation="none", initialize_settings='R', init_scale=0.05)

#model = ModelPFCNoMod(input_shape=(15,), output_shape=(5,), hidden_size=64, output_activation="none", initialize_settings='R', hebbian_type=3, init_scale=0.05)
#model = ModelPFCPreMod(input_shape=(15,), output_shape=(5,), hidden_size=64, output_activation="none", initialize_settings='R', hebbian_type=3, init_scale=0.05)
#model = ModelPFCAfterMod(input_shape=(15,), output_shape=(5,), hidden_size=64, output_activation="none", initialize_settings='R', hebbian_type=3, init_scale=0.05)

#model = ModelPRNNNoMod(input_shape=(15,), output_shape=(5,), hidden_size=64, output_activation="none", initialize_settings='R', hebbian_type=3, init_scale=0.05)
#model = ModelPRNNPreMod(input_shape=(15,), output_shape=(5,), hidden_size=64, output_activation="none", initialize_settings='R', hebbian_type=3, init_scale=0.05)
# Hebbian-Type: 1. alpha ABCD; 2. Eligibility Traces; 3. Decomposed Plasticity; 4. Evolving & Merging
model = ModelPRNNAfterMod(input_shape=(15,), output_shape=(5,), hidden_size=64, output_activation="none", initialize_settings='R', hebbian_type=3, init_scale=0.05)

# Refer to config_SeqPred_task.py for other configurations

#If no load_model is specified, the model is random initialized
#load_model = root + "demo/models/model.maze15_prnn64_aftermod.dat"

#Address for xparl servers, do "xparl start " in your server
server = "localhost:8010"

#True Batch size = Actor_number * batch_size
actor_number = 20
batch_size = 1
task_sub_iterations = 4
inner_rollouts = [(0.0, "TRAIN", True), (0.0, "TRAIN", True), (0.16, "TEST", True),
        (0.22, "TEST", True), (0.36, "TEST", True), (0.64, "TEST", True),
        (0.8, "TEST", True), (1.0, "TEST", True)
        ]


#Select the inner-loop type, for PRNN / RNN / LSTM / EPMLP select "forward", for ES-MAML select "policy_gradient_continuous"
adapt_type = "forward"
ent_factor = 1.0e-6

evolution_pool_size = 80
evolution_topk_size = 40
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

#Sampled Tasks for meta-training
def train_patterns(n_step=0):
    return resample_maze15(n=12)
    #return resample_maze9(n=12)

#Sampled Tasks for meta-testing
def valid_patterns(pattern_number=1024):
    return import_mazes(n=40, file_name="./demo/tasks/1024_maze15.dat")

def game():
    return MazeTask()

def test_patterns():
    return import_mazes(n=2048, file_name="./demo/tasks/2048_maze15.dat")

test_load_model = "./demo/models/maze_decprnn_postdn_demo.dat"
