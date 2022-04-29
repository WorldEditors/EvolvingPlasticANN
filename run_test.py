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
from inner_loop_agents import *
from sklearn.manifold import TSNE
from EStool import ESTool

def get_mean_std(arr):
    tarr = numpy.array(arr)
    exp = numpy.mean(tarr, axis=0)
    std = numpy.mean(tarr * tarr, axis=0) - exp * exp
    l = numpy.shape(tarr)[0]
    return exp, numpy.sqrt(std / l)

def TSNE_trans(weights):
    tsne = TSNE(n_components=2, init='pca', random_state=501)
    x_tsne = tsne.fit_transform(weights)
    return x_tsne

class LocalEvaluator(object):
    def __init__(self, config_file):
        self._config = importlib.import_module(config_file)
        self._nn = self._config.model
        self._output_to_action = self._config.output_to_action 
        self._obs_to_input = self._config.obs_to_input
        self._game = self._config.game()
        self._ent_factor = self._config.ent_factor
        self._adapt_type = self._config.adapt_type

    def cal_score(self, weights_x, shape, static_weights, pattern_list):
        score_rollouts_list = []
        step_rollouts_list = []
        weighted_scores = []
        optimal_steps = []
        uncertainty_list = []
        ent_list = []
        hidden_states = []
        connection_weights = []
        idx = 0
        for pattern in pattern_list:
            idx += 1
            print("Test Pattern : %d" % idx)
            print(weights_x.shape)
            self._nn.from_vector(weights_x, shape)
            self._nn.set_all_parameters(static_weights, is_static=True)
            self._nn.reset()
            additional_wht = self._ent_factor * param_norm_2(weights_x)
            if(self._adapt_type == "forward"):
                #recursion
                weighted_score, score_rollouts, step_rollouts, ext_info = inner_loop_forward(self._config, pattern, self._nn, self._game, additional_wht, is_meta_test=True)
            else:
                raise Exception("No such inner loop type: %s"%self._adapt_type)
            print("Score:%s"%score_rollouts)
            weighted_scores.append(weighted_score)
            score_rollouts_list.append(score_rollouts)
            step_rollouts_list.append(step_rollouts)
            ent_list.append(ext_info["entropy"])
            uncertainty_list.append(ext_info["certainty"])
            if("hidden_states" in ext_info):
                hidden_states.append(ext_info["hidden_states"])
            if("connection_weights" in ext_info):
                connection_weights.append(ext_info["connection_weights"])

            self._game.reset(pattern, "TEST")
            optimal_steps.append(self._game.optimal_steps())
        return score_rollouts_list,  uncertainty_list,  ent_list, hidden_states, connection_weights

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

    def cal_score(self, weights_x, shape, static_weights, pattern_list):
        score_rollouts_list = []
        step_rollouts_list = []
        weighted_scores = []
        optimal_steps = []
        uncertainty_list = []
        ent_list = []
        idx = 0
        for pattern in pattern_list:
            idx += 1
            print("Test Pattern : %d" % idx)
            self._nn.from_vector(weights_x, shape)
            self._nn.set_all_parameters(static_weights, is_static=True)
            self._nn.reset()
            additional_wht = self._ent_factor * param_norm_2(weights_x)
            if(self._adapt_type == "forward"):
                #recursion
                weighted_score, score_rollouts, step_rollouts, ext_info = inner_loop_forward(self._config, pattern, self._nn, self._game, additional_wht, is_meta_test=True)
            else:
                raise Exception("No such inner loop type: %s"%self._adapt_type)
            weighted_scores.append(weighted_score)
            score_rollouts_list.append(score_rollouts)
            step_rollouts_list.append(step_rollouts)
            ent_list.append(ext_info["entropy"])
            uncertainty_list.append(ext_info["certainty"])

            self._game.reset(pattern, "TEST")
            optimal_steps.append(self._game.optimal_steps())

        return score_rollouts_list,  uncertainty_list, ent_list

class RemoteEvaluator(object):
    def __init__(self, config_file):
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
        self._evolution_handler.load(config.test_load_model)

        parl.connect(config.server)
        self._evaluators = [Evaluator(config_file) for _ in range(self._actor_number)]
        self._pattern_retain_iterations = config.pattern_retain_iterations
        self._max_wait_time = 60
        self._max_wait_time_eval = 120
        self._config = config
        self._failed_actors = set()
        self._pattern_kept_time = 0
        print("... Intialization Finished")

    def check_active_actors(self):
        if(len(self._failed_actors) > self._actor_number // 2):
            raise Exception("To many actors failed, quit job, please check your cluster")

    def eval(self):
        tasks = []
        test_pattern_lst = self._config.test_patterns()
        weights = self._evolution_handler._base_weights
        score_rollouts = []
        uncert_lists = []
        ent_lists = []
        deta = (len(test_pattern_lst)  - 1) // (self._actor_number - len(self._failed_actors)) + 1
        i = 0
        j = 0
        wait_time = 0
        unrecv_res = set()
        print("Start Evaluation, Each CPU calculate %d Tasks"%deta)
        while j < len(test_pattern_lst):
            if(i not in self._failed_actors):
                tasks.append(self._evaluators[i].cal_score(
                        weights,
                        self._nn_shape, self._evolution_handler.get_static_weights,
                        test_pattern_lst[j:j+deta])
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
                    score_rollout, uncert_list, ent_list = tasks[idx].get_nowait()
                    score_rollouts.extend(score_rollout)
                    uncert_lists.extend(uncert_list)
                    ent_lists.extend(ent_list)
                except Exception:
                    unrecv_res.add(idx)
            wait_time += 1
            #print("Wait for %d seconds for acquiring results..."%wait_time)
            sys.stdout.flush()
        if(len(unrecv_res) > 0):
            print("Out of time for servers id: %s" % unrecv_res)
            for key in unrecv_res:
                self._failed_actors.add(key)
        print("uncert_lists", numpy.mean(uncert_lists, axis=0))
        print("ent_lists", numpy.mean(ent_lists, axis=0))
        return get_mean_std(score_rollouts)

def local_eval(config):
    ####
    nn = config.model
    evolution_handler = ESTool(
            config.evolution_pool_size, 
            config.evolution_topk_size,
            config.evolution_step_size,
            default_cov_lr=config.evolution_lr,
            segments=nn.para_segments
            )
    para_vec, nn_shape = nn.to_vector
    print("Current Parameter Number: %d, parameters: %s" % (len(para_vec), nn_shape))

    evolution_handler.load(config.test_load_model)

    evaluator = LocalEvaluator(config_module_name)
    tst_pattern_lst = config.test_patterns()
    print("len", len(tst_pattern_lst))

    weights = evolution_handler._base_weights
    # hidden_states & connection_weights: n_pattern * n_rollout * n_step
    scores, uncert_lists, ent_lists, hidden_states, connection_weights = evaluator.cal_score(
            weights,
            nn_shape, evolution_handler.get_static_weights,
            tst_pattern_lst)
    print("uncert_lists", numpy.mean(uncert_lists, axis=0))
    print("entropy_lists", numpy.mean(ent_lists, axis=0))

    # Get the shape_list
    h_weights = []
    w_weights = []
    h_pos_list = []
    w_pos_list = []
    if(len(hidden_states) > 0):
        shapes = []
        for pattern_h in hidden_states:
            shapes.append([])
            for rollout_h in pattern_h:
                shapes[-1].append(len(rollout_h))
                h_weights.extend(rollout_h)
        h_pos = TSNE_trans(h_weights)
        cur_b = 0
        for shape in shapes:
            h_pos_list.append([])
            for rollout_shape in shape:
                cur_e = cur_b + rollout_shape
                h_pos_list[-1].append(h_pos[cur_b:cur_e])
                cur_b = cur_e
    if(len(connection_weights) > 0):
        shapes = []
        for pattern_w in connection_weights:
            shapes.append([])
            for rollout_w in pattern_w:
                shapes[-1].append(len(rollout_w))
                w_weights.extend(rollout_w)
        w_pos = TSNE_trans(w_weights)
        cur_b = 0
        for shape in shapes:
            w_pos_list.append([])
            for rollout_shape in shape:
                cur_e = cur_b + rollout_shape
                w_pos_list[-1].append(w_pos[cur_b:cur_e])
                cur_b = cur_e

    # Perform TSNE visualization
    exp, var = get_mean_std(scores)

    return exp, var, w_pos_list, h_pos_list

def write_trajectory(trajectory, writer):
    idx = 0
    for pattern_traj in trajectory:
        rollout_idx=0
        for rollout_traj in pattern_traj:
            writer.write("Pattern:%d Rollout:%d\n"%(idx, rollout_idx))
            for xy in rollout_traj:
                writer.write("%f %f\n"%(xy[0], xy[1]))
            rollout_idx += 1
        idx += 1


if __name__=='__main__':
    if(len(sys.argv) < 3):
        print("Usage: %s configuration_file is_remote" % sys.argv[0])
        sys.exit(1)

    if(sys.argv[2] == "remote"):
        remote = True
    elif(sys.argv[2] == "local"):
        remote = False
    else:
        raise Exception("Stype can only be remote / local, received %s"%sys.argv[2])

    config_module_name = sys.argv[1].replace(".py", "")
    config = importlib.import_module(config_module_name)
    logger = sys.stdout

    avg_score = - 1.0e+10

    #initial_test
    if(remote):
        Evaluator = RemoteEvaluator(config_module_name)
        exp, var = Evaluator.eval()
        w_traj = []
        h_traj = []
    else:
        exp, var, w_traj, h_traj = local_eval(config)

    logger.write(" ".join(map(lambda x:"%f,%f"%(x[0],x[1]), zip(exp,var))))
    logger.write("\n")

    if(len(w_traj) > 0):
        w_traj_logger = open("w_trajectory.dat", "w")
        write_trajectory(w_traj, w_traj_logger)
        w_traj_logger.close()
    if(len(h_traj) > 0):
        h_traj_logger = open("h_trajectory.dat", "w")
        write_trajectory(h_traj, h_traj_logger)
        h_traj_logger.close()
