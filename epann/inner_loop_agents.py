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
    gamma = 0.96
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
        ext_info["adapt_nc_rollout"] = []
        nc_exist = True
    else:
        nc_exist = False
    if(is_meta_test and ("mem" in learner.l1.__dict__ or "mem_h" in  learner.l1.__dict__)):
        ext_info["hidden_states"] = []
        ext_info["adapt_h_step"] = []
        ext_info["adapt_h_rollout"] = []
        h_exist = True
    else:
        h_exist = False

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

        if(nc_exist):
            ext_info["connection_weights"].append([])
            d_nc_steps = []
            d_nc_rollouts = []
            if("heb_h" in learner.l1.__dict__):
                cur_nc = numpy.ravel(learner.l1.heb_h())
            if("heb" in learner.l2.__dict__):
                cur_nc = numpy.ravel(learner.l2.heb())
            if("mem_c" in learner.l1.__dict__):
                cur_nc = numpy.ravel(learner.l1.mem_c())
            if(rollout_idx == 1):
                nc_ema = numpy.copy(cur_nc)
                prev_ema_nc = numpy.copy(nc_ema)
                init_nc_ema = numpy.copy(cur_nc)
        if(h_exist):
            ext_info["hidden_states"].append([])
            d_h_rollouts = []
            if("mem" in learner.l1.__dict__):
                cur_h = numpy.ravel(learner.l1.mem())
            if("mem_h" in learner.l1.__dict__):
                cur_h = numpy.ravel(learner.l1.mem_h())
            if(rollout_idx == 1):
                h_ema = numpy.copy(cur_h)
                prev_ema_h = numpy.copy(h_ema)
                init_h_ema = numpy.copy(cur_h)

        inputs, _ = config.obs_to_input(obs, action, info)
        steps = 1
        decisive_steps = 0
        avg_ent = 0
        is_goal = 0
        while not done:
            # Only do hebbian when train episode
            output = learner(inputs)
            if(nc_exist):
                if("heb_h" in learner.l1.__dict__):
                    cur_nc = numpy.ravel(learner.l1.heb_h())
                if("heb" in learner.l2.__dict__):
                    cur_nc = numpy.ravel(learner.l2.heb()) 
                if("mem_c" in learner.l1.__dict__):
                    cur_nc = numpy.ravel(learner.l1.mem_c())
                ext_info["connection_weights"][-1].append(cur_nc)

                nc_ema = cur_nc * (1.0 - gamma) + gamma * nc_ema
                d_nc = numpy.linalg.norm(nc_ema - cur_nc)
                d_nc_rollouts.append(d_nc)

            if(h_exist):
                if("mem" in learner.l1.__dict__):
                    cur_h = numpy.ravel(learner.l1.mem())
                if("mem_h" in learner.l1.__dict__):
                    cur_h = numpy.ravel(learner.l1.mem_h())
                ext_info["hidden_states"][-1].append(cur_h)

                h_ema = cur_h * (1.0 - gamma) + gamma * h_ema
                d_h = numpy.linalg.norm(h_ema - cur_h)
                d_h_rollouts.append(d_h)

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
            if(nc_exist):
                ext_info["adapt_nc_rollout"].append(numpy.linalg.norm(nc_ema - init_nc_ema))
                ext_info["adapt_nc_step"].append(numpy.mean(d_nc_rollouts))
                prev_ema_nc = numpy.copy(nc_ema)
            if(h_exist):
                ext_info["adapt_h_rollout"].append(numpy.linalg.norm(h_ema - init_h_ema))
                ext_info["adapt_h_step"].append(numpy.mean(d_h_rollouts))
                prev_ema_h = numpy.copy(h_ema)
        weights_all += rollout_weight
        rollout_num += 1

    if(is_meta_test):
        if(nc_exist):
            ext_info["adapt_nc_end"] = numpy.linalg.norm(nc_ema - init_nc_ema)
        if(h_exist):
            ext_info["adapt_h_end"] = numpy.linalg.norm(h_ema - init_h_ema)
    return score / weights_all, score_rollouts, step_rollouts, ext_info
