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

from epann.utils import make_dir
from epann.utils import add_params, diff_params, multiply_params, mean_params, sum_params, param_max
from epann.inner_loop_agents import *
from epann.EStool import ESTool

@parl.remote_class(wait=False)
class Evaluator(object):
    def __init__(self, config_file):
        self._config = importlib.import_module(config_file)
        self._nn = self._config.model
        self._output_to_action = self._config.output_to_action 
        self._obs_to_input = self._config.obs_to_input
        self._game = self._config.game()
        self._ent_factor = self._config.ent_factor
        self._adapt_type = self._config.adapt_type

    def cal_score(self, i, weights_x, shape, static_weights, pattern_list, task_iterations):
        score_rollouts_list = []
        step_rollouts_list = []
        weighted_scores = []
        for _ in range(task_iterations):
            for pattern in pattern_list:
                #raise Exception(pattern)
                self._nn.from_vector(weights_x, shape)
                self._nn.set_all_parameters(static_weights, is_static=True)
                additional_wht = self._ent_factor * param_norm_2(weights_x)
                self._nn.reset()
                if(self._adapt_type == "forward"):
                    #recursion
                    weighted_score, score_rollouts, step_rollouts, _ = inner_loop_forward(self._config, pattern, self._nn, self._game, additional_wht)
                else:
                    raise Exception("No such inner loop type: %s"%self._adapt_type)
                weighted_scores.append(weighted_score)
                score_rollouts_list.append(score_rollouts)
                step_rollouts_list.append(step_rollouts)
        return i, numpy.mean(weighted_scores), numpy.mean(score_rollouts_list, axis=0), numpy.mean(step_rollouts_list, axis=0), param_max(weights_x)

class Trainer(object):
    def __init__(self, config_file):
        print("... Intializing evolution pool")
        config = importlib.import_module(config_file)
        self._actor_number = config.actor_number
        self._nn = config.model
        self._evolution_handler = ESTool(
                config.evolution_pool_size, 
                config.evolution_topk_size,
                config.evolution_step_size,
                default_cov_lr=config.evolution_lr,
                segments=self._nn.para_segments
                )
        para_vec, self._nn_shape = self._nn.to_vector
        print("Current Parameter Number: %d, parameters: %s" % (len(para_vec), self._nn_shape))
        print("Current Static Parameter parameters: %s" % (self._nn.static_parameters.keys()))
        if("load_model" in config.__dict__):
            self._evolution_handler.load(config.load_model)
        else:
            self._evolution_handler.init_popultation(para_vec, self._nn.static_parameters)
        print(self._nn.static_parameters.keys())

        parl.connect(config.server, distributed_files=['./epann/*.py', './envs/*.py'])
        self._evaluators = [Evaluator(config_file) for _ in range(self._actor_number)]
        self._pattern_list = config.train_patterns()
        self._task_iterations = config.task_sub_iterations
        self._max_wait_time = 120
        self._max_wait_time_eval = 240
        self._config = config
        self._failed_actors = set()
        self._pattern_kept_time = 0
        print("... Intialization Finished")

    def check_active_actors(self):
        if(len(self._failed_actors) > self._actor_number // 2):
            raise Exception("To many actors failed, quit job, please check your cluster")

    def evolve(self, iteration):
        self._pattern_list = self._config.train_patterns(n_step=iteration)
        tasks = []
        
        #distribute tasks
        i = 0
        score_rollouts = [[] for i in range(self._evolution_handler.pool_size)]
        step_rollouts = [[] for i in range(self._evolution_handler.pool_size)]
        failed_res = set()
        norms = []
        while i < self._evolution_handler.pool_size:
            unrecv_res = dict()
            i_b = i
            for j in range(self._actor_number):
                if(i < self._evolution_handler.pool_size and j not in self._failed_actors):
                    tasks.append((j, self._evaluators[j].cal_score(i,
                            self._evolution_handler.get_weights(i), 
                            self._nn_shape, self._evolution_handler.get_static_weights,
                            self._pattern_list, 
                            self._task_iterations)))
                    unrecv_res[i] = j
                    i+=1
            i_e = i
            wait_time = 0
            while wait_time < self._max_wait_time and len(unrecv_res) > 0:
                print("unrecv_res:", unrecv_res)
                time.sleep(1)
                remove_keys = set()
                for key in unrecv_res:
                    try:
                        cur_i, weighted_score, score_rollout, step_rollout, norm = tasks[key][1].get_nowait()
                        self._evolution_handler.set_score(cur_i, weighted_score)
                        score_rollouts[cur_i] = score_rollout
                        step_rollouts[cur_i] = step_rollout
                        norms.append(norm)
                        remove_keys.add(key)
                    except Exception:
                        pass
                for key in remove_keys:
                    del unrecv_res[key]
                wait_time += 1
                #print("Wait for %d seconds for acquiring results..."%wait_time)
                sys.stdout.flush()
            if(len(unrecv_res) > 0):
                for key in unrecv_res.keys():
                    print("remote server idx %d exceed time limits, abandom the server" % unrecv_res[key])
                    self._failed_actors.add(unrecv_res[key])
                    failed_res.add(key)

        for cur_i in sorted(failed_res, reverse=True):
            del self._evolution_handler._evolution_pool[cur_i]
            del score_rollouts[cur_i]
            del step_rollouts[cur_i]

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
        valid_pattern_lst = self._config.valid_patterns()
        weights = self._evolution_handler._base_weights
        score_rollouts = []
        step_rollouts = []
        whts = []
        deta = (len(valid_pattern_lst)  - 1) // (self._actor_number - len(self._failed_actors)) + 1
        i = 0
        j = 0
        wait_time = 0
        unrecv_res = set()
        print("Start Evaluation, Each CPU calculate %d Tasks"%deta)
        while j < len(valid_pattern_lst):
            if(i not in self._failed_actors):
                tasks.append(self._evaluators[i].cal_score(i,
                        weights,
                        self._nn_shape, self._evolution_handler.get_static_weights,
                        valid_pattern_lst[j:j+deta],
                        self._task_iterations)
                        )
                j += deta
                unrecv_res.add(i)
            i += 1

        #print("Waiting for acquiring results...")
        while wait_time < self._max_wait_time_eval and len(unrecv_res) > 0:
            time.sleep(1)
            for _ in range(len(unrecv_res)):
                idx = unrecv_res.pop()
                try:
                    cur_i, weighted_score, score_rollout, step_rollout, norm = tasks[idx].get_nowait()
                    score_rollouts.append(score_rollout)
                    step_rollouts.append(step_rollout)
                    whts.append(weighted_score)
                except Exception:
                    unrecv_res.add(idx)
            wait_time += 1
            #print("Wait for %d seconds for acquiring results..."%wait_time)
            sys.stdout.flush()
        if(len(unrecv_res) > 0):
            print("Out of time for servers id: %s" % unrecv_res)
            for key in unrecv_res:
                self._failed_actors.add(key)

        n = numpy.shape(score_rollouts)[0]
        score1_rollouts = numpy.mean(numpy.array(score_rollouts), axis=0)
        score2_rollouts = numpy.mean(numpy.array(score_rollouts) * numpy.array(score_rollouts), axis=0)
        var_rollouts = numpy.sqrt(numpy.clip(score2_rollouts - score1_rollouts * score1_rollouts, 0.0, 1.0e+10) / n)

        step_rollouts = numpy.mean(numpy.array(step_rollouts), axis=0)
        whts = numpy.mean(whts)
        return whts, score1_rollouts, step_rollouts, var_rollouts

    def save(self, filename):
        self._evolution_handler.save(filename)

if __name__=='__main__':
    if(len(sys.argv) < 2):
        print("Usage: %s configuration_file" % sys.argv[0])
        sys.exit(1)
    config_module_name = sys.argv[1].replace(".py", "")
    config = importlib.import_module(config_module_name)
    log_file = config.directory + "/train.log"
    model_directory = config.directory + "/models/"
    make_dir(model_directory)
    logger = open(log_file, "a")

    smooth_scores = []
    avg_score = - 1.0e+10
    trainer = Trainer(config_module_name)

    #initial_test
    wht, scores, steps, var = trainer.eval()
    localtime = time.asctime(time.localtime(time.time()))
    logger.write("%s\tTest_Record\tCurrent_Iteration:%5d\tWeighted_Score:%f\tRollout_Score:%s\tRollout_Step:%s\tRollout_Var:%s\n"%
            (localtime, 1, wht, ",".join(map(str, scores)), ",".join(map(str, steps)), ",".join(map(str, var))))

    for iteration in range(config.max_iter):
        top_5_score, score_rollouts, step_rollouts, norm = trainer.evolve(iteration)
        if((iteration + 1) % config.save_iter == 0):
            trainer.save(model_directory + "model.%06d.dat"%(iteration+1))
            print("Models model.%06d.dat saved"%(iteration+1))
        localtime = time.asctime(time.localtime(time.time()))
        smooth_scores.append(top_5_score)
        smooth_scores = smooth_scores[-100:]
        avg_score = numpy.mean(smooth_scores) 
        logger.write("%s iteration: %d; smoothed_score: %.4f; score_rollouts: %s; step_roullouts: %s; parameter - l2norm: %.4f; top_k_score: %.4f\n"%
                (localtime, iteration + 1, avg_score, score_rollouts, step_rollouts, norm, top_5_score))
        logger.flush()
        if((iteration + 1) % config.test_iter == 0):
            wht, scores, steps, var = trainer.eval()
            logger.write("%s\tTest_Record\tCurrent_Iteration:%5d\tWeighted_Score:%f\tRollout_Score:%s\tRollout_Step:%s\tRollout_Var:%s\n"%
                    (localtime, iteration + 1, wht, ",".join(map(str, scores)), ",".join(map(str, steps)), ",".join(map(str, var))))
    logger.close()
