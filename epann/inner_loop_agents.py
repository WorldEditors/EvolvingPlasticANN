"""
Inner Loops for Generalized Tasks
get_adaption_score_bp: Back-Propagation for Supervised Tasks
get_adaption_score_rec: Recursion + Plasticity Learning
get_adpation_score_pg: Policy Gradient for RL Tasks (continuous action space)
get_adpation_score_discrete_pg: Policy Gradient for RL Tasks (discrete action space)
"""
import numpy
from numpy import random
from epann.utils import param_norm, param_norm_2, param_max

def str_list(lst):
    return "\t".join(map(str, lst))

# back propagation (supervised learning) in inner-loop
def inner_loop_forward(config, pattern, learner, game, additional_wht, is_meta_test=False):
    score = 0.0

    weights_all = 0.0
    score_rollouts = []
    step_rollouts = []
    rollout_num = 0
    ext_info = dict()
    ext_info["exploration"] = []
    if(is_meta_test):
        ext_info["certainty"] = []
        ext_info["goal_arr"] = []
        ext_info["entropy"] = []
    if(is_meta_test and ("heb" in learner.l2.__dict__ or "mem_c" in learner.l1.__dict__ or "heb_h" in learner.l1.__dict__)):
        ext_info["connection_weights"] = []
        ext_info["adapt_nc_step"] = []
    if(is_meta_test and ("mem" in learner.l1.__dict__ or "mem_h" in  learner.l1.__dict__)):
        ext_info["hidden_states"] = []
        ext_info["adapt_h_step"] = []
    rollout_idx = 0
    game.task_reset()
    for rollout_weight, eps_type, is_test in config.inner_rollouts:
        rollout_idx += 1
        done = False
        obs = game.reset(pattern, eps_type)
        action = game.default_action()
        info = game.default_info()
        info["test"] = is_test
        info["rollout"] = rollout_idx

        if(is_meta_test and ("heb" in learner.l2.__dict__ or "mem_c" in learner.l1.__dict__ or "heb_h" in learner.l1.__dict__)):
            ext_info["connection_weights"].append([])
            if(is_meta_test and "heb_h" in learner.l1.__dict__):
                nc_start = numpy.ravel(learner.l1.heb_h())
            if(is_meta_test and "heb" in learner.l2.__dict__):
                nc_start = numpy.ravel(learner.l2.heb())
            if(is_meta_test and "mem_c" in learner.l1.__dict__):
                nc_start = numpy.ravel(learner.l1.mem_c())
            nc_end = nc_start
        if(is_meta_test and ("mem" in learner.l1.__dict__ or "mem_h" in  learner.l1.__dict__)):
            ext_info["hidden_states"].append([])
            if(is_meta_test and "mem" in learner.l1.__dict__):
                h_start = numpy.ravel(learner.l1.mem())
            if(is_meta_test and "mem_h" in learner.l1.__dict__):
                h_start = numpy.ravel(learner.l1.mem_h())
            h_end = h_start

        inputs, _ = config.obs_to_input(obs, action, info)
        steps = 1
        decisive_steps = 0
        avg_ent = 0
        is_goal = 0
        while not done:
            # Only do hebbian when train episode
            output = learner(inputs)
            if(is_meta_test and "heb_h" in learner.l1.__dict__):
                prev_nc_end = nc_end
                nc_end = numpy.ravel(learner.l1.heb_h())
                ext_info["connection_weights"][-1].append(nc_end)
                ext_info["adapt_nc_step"].append(numpy.linalg.norm(prev_nc_end - nc_end))
            if(is_meta_test and "heb" in learner.l2.__dict__):
                prev_nc_end = nc_end
                nc_end = numpy.ravel(learner.l2.heb()) 
                ext_info["connection_weights"][-1].append(nc_end)
                ext_info["adapt_nc_step"].append(numpy.linalg.norm(prev_nc_end - nc_end))
            if(is_meta_test and "mem_c" in learner.l1.__dict__):
                prev_nc_end = nc_end
                nc_end = numpy.ravel(learner.l1.mem_c())
                ext_info["connection_weights"][-1].append(nc_end)
                ext_info["adapt_nc_step"].append(numpy.linalg.norm(prev_nc_end - nc_end))
            if(is_meta_test and "mem" in learner.l1.__dict__):
                prev_h_end = h_end
                h_end = numpy.ravel(learner.l1.mem())
                ext_info["hidden_states"][-1].append(h_end)
                ext_info["adapt_h_step"].append(numpy.linalg.norm(prev_h_end - h_end))
            if(is_meta_test and "mem_h" in learner.l1.__dict__):
                prev_h_end = h_end
                h_end = numpy.ravel(learner.l1.mem_h())
                ext_info["hidden_states"][-1].append(h_end)
                ext_info["adapt_h_step"].append(numpy.linalg.norm(prev_h_end - h_end))
            action, act_info = config.output_to_action(output, info)
            if("argmax" in act_info and act_info["argmax"]):
                decisive_steps += 1
            done, obs, info = game.step(action)
            info["rollout"] = rollout_idx
            info["test"] = is_test
            if(info["goal"]):
                is_goal = 1
            inputs, _ = config.obs_to_input(obs, action, info)
            if("entropy" in act_info):
                avg_ent += act_info["entropy"]

            #if(is_meta_test):
                #print(rollout_idx, obs, info)
                
            steps += 1
        #put the information of the last step in it
        score += (game.score - additional_wht) * rollout_weight
        score_rollouts.append(game.score)
        step_rollouts.append(steps)
        ext_info["exploration"].append(game.coverage_rate())
        if(is_meta_test):
            ext_info["certainty"].append(decisive_steps / (steps - 1))
            ext_info["entropy"].append(avg_ent / (steps - 1))
            ext_info["goal_arr"].append(is_goal)
            if("adapt_nc_step" in ext_info):
                ext_info["adapt_nc_end"] = numpy.linalg.norm(nc_end - nc_start)
            if("adapt_h_step" in ext_info):
                ext_info["adapt_h_end"] = numpy.linalg.norm(h_end - h_start)
        weights_all += rollout_weight
        rollout_num += 1
    return score / weights_all, score_rollouts, step_rollouts, ext_info
