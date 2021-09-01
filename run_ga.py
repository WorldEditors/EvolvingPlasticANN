"""
Runing Meta-Training and Meta-Testing Loops
"""
import sys
import time
import numpy
import parl
import importlib
from time import sleep
from copy import copy, deepcopy 
from utils import make_dir
from utils import add_params, diff_params, multiply_params, mean_params, sum_params, param_max
from inner_loop import get_adaption_score_bp, get_adaption_score_pg, get_adaption_score_discrete_pg, get_adaption_score_rec
from EStool import ESTool
from neural_network import PlasticNN

@parl.remote_class(wait=False)
class Evaluator(object):
    def __init__(self, config_file):
        self._config = importlib.import_module(config_file)
        self._nn = PlasticNN(
                input_neurons=self._config.input_neurons, 
                model_structures=self._config.model_structures)
        self._output_to_action = self._config.output_to_action 
        self._obs_to_input = self._config.obs_to_input
        self._game = self._config.game()
        self._ent_factor = self._config.ent_factor
        self._adapt_type = self._config.adapt_type

    def cal_score(self, i, weights_x, pattern_list, task_iterations):
        score_rollouts_list = []
        step_rollouts_list = []
        weighted_scores = []
        for _ in range(task_iterations):
            for pattern in pattern_list:
                if(self._adapt_type == "bp"):
                    #gradient descent
                    weighted_score, score_rollouts, step_rollouts = get_adaption_score_bp(self._config, pattern, self._nn, self._game, weights_x, self._ent_factor)
                elif(self._adapt_type == "pg"):
                    #policy gradient
                    weighted_score, score_rollouts, step_rollouts = get_adaption_score_pg(self._config, pattern, self._nn, self._game, weights_x, self._ent_factor)
                elif(self._adapt_type == "heb"):
                    #hebb's rule
                    weighted_score, score_rollouts, step_rollouts = get_adaption_score_heb(self._config, pattern, self._nn, self._game, weights_x, self._ent_factor)
                elif(self._adapt_type == "recursive"):
                    #recursion
                    weighted_score, score_rollouts, step_rollouts = get_adaption_score_rec(self._config, pattern, self._nn, self._game, weights_x, self._ent_factor)
                elif(self._adapt_type == "discrete_pg"):
                    #recursion
                    weighted_score, score_rollouts, step_rollouts = get_adaption_score_discrete_pg(self._config, pattern, self._nn, self._game, weights_x, self._ent_factor)
                weighted_scores.append(weighted_score)
                score_rollouts_list.append(score_rollouts)
                step_rollouts_list.append(step_rollouts)
        return i, numpy.mean(weighted_scores), numpy.mean(score_rollouts_list, axis=0), numpy.mean(step_rollouts_list, axis=0), param_max(weights_x)

class Trainer(object):
    def __init__(self, config_file):
        print("... Intializing evolution pool")
        config = importlib.import_module(config_file)
        self._actor_number = config.actor_number
        self._model = PlasticNN(
                input_neurons=config.input_neurons,
                model_structures=config.model_structures)
        self._evolution_handler = ESTool(
                config.evolution_pool_size, 
                self._model._noise_factor,
                config.learning_rate)
        if("load_model" in config.__dict__):
            self._evolution_handler.load(config.load_model, config.model_structures)
        else:
            self._evolution_handler.init_popultation(self._model._parameter_list)

        parl.connect(config.server)
        self._evaluators = [Evaluator(config_file) for _ in range(self._actor_number)]
        self._pattern_list = config.gen_pattern()
        self._pattern_renew = config.pattern_renew
        self._pattern_retain_iterations = config.pattern_retain_iterations
        self._task_iterations = config.task_sub_iterations
        self._config = config
        self._pattern_kept_time = 0
        print("... Intialization Finished")

    def evolve(self, iteration):
        self._pattern_kept_time += 1
        if(self._pattern_kept_time >= self._pattern_retain_iterations): 
            #self._pattern_list.extend(self._config.gen_pattern(self._pattern_renew))
            #del(self._pattern_list[:self._pattern_renew])
            self._pattern_list = self._config.gen_pattern()
            self._pattern_kept_time = 0
        tasks = []
        
        #distribute tasks
        i = 0
        score_rollouts = [[] for i in range(self._evolution_handler.pool_size)]
        step_rollouts = [[] for i in range(self._evolution_handler.pool_size)]
        norms = []
        while i < self._evolution_handler.pool_size:
            i_b = i
            for j in range(self._actor_number):
                if(i < self._evolution_handler.pool_size):
                    tasks.append(self._evaluators[j].cal_score(i,
                            self._evolution_handler.get_weights(i), 
                            self._pattern_list, 
                            self._task_iterations))
                    i+=1
            i_e = i
            for idx in range(i_b, i_e):
                cur_i, weighted_score, score_rollout, step_rollout, norm = tasks[idx].get()
                self._evolution_handler.set_score(cur_i, weighted_score)
                score_rollouts[cur_i] = score_rollout
                step_rollouts[cur_i] = step_rollout
                norms.append(norm)
        weighted_score = self._evolution_handler.stat_avg()
        score_rollouts = numpy.mean(numpy.array(score_rollouts), axis=0)
        step_rollouts = numpy.mean(numpy.array(step_rollouts), axis=0)
        localtime = time.asctime(time.localtime(time.time()))
        is_reseted = self._evolution_handler.evolve(verbose=True)
        if(is_reseted):
            self._pattern_kept_time = 0
        #collect results
        return weighted_score, score_rollouts, step_rollouts, numpy.mean(norms)

    def eval(self):
        tasks = []
        test_pattern_lst = self._config.test_patterns()
        weights = self._evolution_handler._cur_param
        score_rollouts = []
        step_rollouts = []
        whts = []
        deta = (len(test_pattern_lst)  - 1) // self._actor_number + 1
        i = 0
        j = 0
        while j < len(test_pattern_lst):
            tasks.append(self._evaluators[i].cal_score(i,
                    weights,
                    test_pattern_lst[j:j+deta],
                    self._task_iterations)
                    )
            j += deta
            i += 1
        for idx in range(len(tasks)):
            cur_i, weighted_score, score_rollout, step_rollout, norm = tasks[idx].get()
            score_rollouts.append(score_rollout)
            step_rollouts.append(step_rollout)
            whts.append(weighted_score)
        score_rollouts = numpy.mean(numpy.array(score_rollouts), axis=0)
        step_rollouts = numpy.mean(numpy.array(step_rollouts), axis=0)
        whts = numpy.mean(whts)
        return whts, score_rollouts, step_rollouts
        

    def save(self, filename):
        self._evolution_handler.save(filename)

if __name__=='__main__':
    if(len(sys.argv) < 2):
        print("Usage: %s configuration_file" % sys.argv[0])
        sys.exit(1)
    config = importlib.import_module(sys.argv[1])
    log_file = config.directory + "/train.log"
    model_directory = config.directory + "/models/"
    make_dir(model_directory)
    logger = open(log_file, "w")

    smooth_scores = []
    avg_score = - 1.0e+10
    trainer = Trainer(sys.argv[1])

    #initial_test
    wht, scores, steps = trainer.eval()
    localtime = time.asctime(time.localtime(time.time()))
    logger.write("%s\tTest_Record\tCurrent_Iteration:%5d\tWeighted_Score:%f\tRollout_Score:%s\tRollout_Step:%s\n"%
	    (localtime, 0, wht, ",".join(map(str, scores)), ",".join(map(str, steps))))

    for iteration in range(config.max_iter):
        top_5_score, score_rollouts, step_rollouts, norm = trainer.evolve(iteration)
        if((iteration + 1) % config.save_iter == 0):
            trainer.save(model_directory + "model.%06d.dat"%(iteration+1))
        localtime = time.asctime(time.localtime(time.time()))
        smooth_scores.append(top_5_score)
        smooth_scores = smooth_scores[-100:]
        avg_score = numpy.mean(smooth_scores) 
        logger.write("%s iteration: %d; smoothed_score: %.4f; score_rollouts: %s; step_roullouts: %s; parameter - l2norm: %s; top_k_score: %.4f\n"%
                (localtime, iteration + 1, avg_score, score_rollouts, step_rollouts, norm, top_5_score))
        logger.flush()
        if((iteration + 1) % config.test_iter == 0):
            wht, scores, steps = trainer.eval()
            logger.write("%s\tTest_Record\tCurrent_Iteration:%5d\tWeighted_Score:%f\tRollout_Score:%s\tRollout_Step:%s\n"%
                    (localtime, iteration + 1, wht, ",".join(map(str, scores)), ",".join(map(str, steps))))
    logger.close()
