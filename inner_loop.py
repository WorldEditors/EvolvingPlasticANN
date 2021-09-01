"""
Inner Loops for Generalized Tasks
get_adaption_score_bp: Back-Propagation for Supervised Tasks
get_adaption_score_rec: Recursion + Plasticity Learning
get_adpation_score_pg: Policy Gradient for RL Tasks (continuous action space)
get_adpation_score_discrete_pg: Policy Gradient for RL Tasks (discrete action space)
"""
import numpy
from numpy import random
from neural_network import PlasticNN
from utils import param_norm, param_norm_2, param_max

def str_list(lst):
    return "\t".join(map(str, lst))

# back propagation (supervised learning) in inner-loop
def get_adaption_score_rec(config, pattern, nn, game, weights, norm_factor):
    nn.set_params(weights)
    w_norm = param_norm_2(weights)
    score = 0.0

    weights_all = 0.0
    score_rollouts = []
    step_rollouts = []
    rollout_num = 0
    tmp_ret_rnn = []
    for rollout_weight, eps_type, is_train in config.inner_rollouts:
        done = False
        steps = 0.0
        start_obs = game.reset(pattern, eps_type)
        if(is_train):
            use_hebb = True
        else:
            use_hebb = False
        default_action = game.default_action()
        default_info = game.default_info()
        inputs = config.obs_to_input(start_obs, default_action, default_info, is_train)
        while not done:
            # Only do hebbian when train episode
            output, _ = nn.run_inference_single(inputs, hebb=use_hebb)
            action = config.output_to_action(output, is_train)
            done, obs, info = game.step(action)
            inputs = config.obs_to_input(obs, action, info, is_train)
            steps += 1.0
        #put the information of the last step in it
        score += (game.score - norm_factor * w_norm) * rollout_weight
        score_rollouts.append(game.score)
        step_rollouts.append(steps)
        weights_all += rollout_weight
        rollout_num += 1
    return score / weights_all, score_rollouts, step_rollouts#, tmp_ret_rnn

# back propagation (supervised learning) in inner-loop
def get_adaption_score_pg(config, pattern, nn, game, weights, norm_factor):
    nn.set_params(weights)
    w_norm = param_norm_2(weights)
    score = 0.0

    weights_all = 0.0
    score_rollouts = []
    step_rollouts = []
    observations = []
    actions = []
    advantage = []
    for rollout_weight, eps_type, is_train in config.inner_rollouts:
        done = False
        steps = 0.0
        start_obs = game.reset(pattern, eps_type)
        default_action = game.default_action()
        default_info = game.default_info()
        tmp_rewards = []
        inputs = config.obs_to_input(start_obs, default_action, default_info, pattern, is_train, extra_info=steps)
        while not done:
            output, _ = nn.run_inference_single(inputs)
            if(eps_type=="TRAIN"):
                true_output = numpy.clip(output + random.normal(size=output.shape,loc=0,scale=0.10), -1.0, 1.0)
                action = config.output_to_action(true_output)
            else:
                action = config.output_to_action(output)
            done, obs, info = game.step(action)
            if(eps_type=="TRAIN"):
                observations.append(inputs)
                actions.append(action)
                tmp_rewards.append(info["reward"])
            inputs = config.obs_to_input(obs, action, info, pattern, is_train, extra_info=steps)
            steps += 1.0
        if(is_train):
            # To correctly calculate the final state
            advantage += config.rewards2adv(tmp_rewards)
        score += (game.score - norm_factor * w_norm) * rollout_weight
        score_rollouts.append(game.score)
        step_rollouts.append(steps)
        weights_all += rollout_weight

        # Do Training
        if(is_train):
            obs_arr = numpy.array(observations)
            action_arr = numpy.array(actions)
            adv_arr = numpy.array(advantage) - numpy.mean(advantage)
            #print(advantage, adv_arr)
            for lr in nn._parameter_list["Inner_LR"]:
                nn.run_pg(lr, obs_arr, action_arr, adv_arr)
    return score / weights_all, score_rollouts, step_rollouts

# back propagation (supervised learning) in inner-loop
def get_adaption_score_bp(config, pattern, nn, game, weights, norm_factor):
    nn.set_params(weights)
    w_norm = param_norm_2(weights)
    score = 0.0

    weights_all = 0.0
    score_rollouts = []
    step_rollouts = []
    features = []
    labels = []
    for rollout_weight, eps_type, is_train in config.inner_rollouts:
        done = False
        steps = 0.0
        start_obs = game.reset(pattern, eps_type)
        default_action = game.default_action()
        default_info = game.default_info()
        inputs, outputs = config.obs_to_input(start_obs, default_action, default_info, is_train)
        features.append(inputs)
        labels.append(outputs)
        while not done:
            output, _ = nn.run_inference_single(inputs)
            action = config.output_to_action(output)
            done, obs, info = game.step(action)
            inputs, outputs = config.obs_to_input(obs, action, info, is_train)
            if(is_train):
                features.append(inputs)
                labels.append(outputs)
            steps += 1.0
        score += (game.score - norm_factor * w_norm) * rollout_weight
        score_rollouts.append(game.score)
        step_rollouts.append(steps)
        weights_all += rollout_weight

        # Do Training
        if(is_train):
            feature_arr = numpy.array(features)
            label_arr = numpy.array(labels)
            for lr in nn._parameter_list["Inner_LR"]:
                nn.run_bp(lr, feature_arr, label_arr)
    return score / weights_all, score_rollouts, step_rollouts

# back propagation (supervised learning) in inner-loop
def get_adaption_score_discrete_pg(config, pattern, nn, game, weights, norm_factor):
    nn.set_params(weights)
    w_norm = param_norm_2(weights)
    score = 0.0

    weights_all = 0.0
    score_rollouts = []
    step_rollouts = []
    observations = []
    actions = []
    advantage = []
    for rollout_weight, eps_type, is_train in config.inner_rollouts:
        done = False
        steps = 0.0
        start_obs = game.reset(pattern, eps_type)
        default_action = game.default_action()
        default_info = game.default_info()
        tmp_rewards = []
        inputs = config.obs_to_input(start_obs, default_action, default_info, is_train)
        while not done:
            output, _ = nn.run_inference_single(inputs)
            action = config.output_to_action(output)
            done, obs, info = game.step(action)
            if(is_train):
                observations.append(inputs)
                actions.append(action)
                tmp_rewards.append(info["reward"])
            inputs = config.obs_to_input(obs, action, info)
            steps += 1.0
        if(is_train):
            # To correctly calculate the final state
            advantage += config.rewards2adv(tmp_rewards)
        score += (game.score - norm_factor * w_norm) * rollout_weight
        score_rollouts.append(game.score)
        step_rollouts.append(steps)
        weights_all += rollout_weight

        # Do Training
        if(is_train):
            obs_arr = numpy.array(observations)
            action_arr = numpy.array(actions)
            adv_arr = numpy.array(advantage) - numpy.mean(advantage)
            for lr in nn._parameter_list["Inner_LR"]:
                nn.run_discrete_pg(lr, obs_arr, action_arr, adv_arr)
    return score / weights_all, score_rollouts, step_rollouts
