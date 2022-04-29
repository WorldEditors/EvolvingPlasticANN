"""
Runing Meta-Training and Meta-Testing Loops
"""
import sys
import time
import numpy
import parl
import importlib
from numpy import random
from time import sleep
from copy import copy, deepcopy 
from utils import make_dir
from utils import add_params, diff_params, multiply_params, mean_params, sum_params, param_max
from inner_loop_agents import *
from EStool import ESTool
from gen_train_test_patterns import gen_patterns
from env_maze import  MazeTask

if __name__=='__main__':
    if(len(sys.argv) < 2):
        print("Usage: %s test_patterns" % sys.argv[0])
        sys.exit(1)
    cell_scale = 11
    pattern_list = gen_patterns(1024, cell_scale, file_name=sys.argv[1])
    logger = sys.stdout

    smooth_scores = []
    avg_score = - 1.0e+10

    #initial_test
    localtime = time.asctime(time.localtime(time.time()))
    optimal_scores = []
    random_scores = []

    ####
    game = MazeTask()
    for pattern in pattern_list:
        game.reset(pattern, "TEST")
        optimal_scores.append(1.0 - 0.01 * game.optimal_steps())
        done = False
        while not done:
            # Only do hebbian when train episode
            action = random.randint(0,4)
            done, obs, info = game.step(action)
        #put the information of the last step in it
        random_scores.append(game.score)
    random_scores = numpy.array(random_scores)
    optimal_scores = numpy.array(optimal_scores)
    exp_rand = numpy.mean(random_scores)
    exp_opt = numpy.mean(optimal_scores)
    l = len(pattern_list)
    var_rand = numpy.sqrt((numpy.mean(random_scores * random_scores) - exp_rand * exp_rand) / l)
    var_opt = numpy.sqrt((numpy.mean(optimal_scores * optimal_scores) - exp_opt * exp_opt) / l)

    logger.write("Random: %f +- %f, Optimal: %f +- %f\n"%(exp_rand, var_rand, exp_opt, var_opt))
