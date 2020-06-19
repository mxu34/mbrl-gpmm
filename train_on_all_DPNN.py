import time
import datetime
import copy
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
from loguru import logger
import yaml

import gym
from utils import dumb_reward_plot, dumb_assignment_predict
from baselines.dpnn.DPNNMM import DPNNMM
from baselines.dpnn.NN import NNComponent

ENVS = 'Cartpole'

if ENVS == 'Cartpole':
    sys.path.append('./envs/cartpole-envs')
    import cartpole_envs
    from mpc.mpc_cp import MPC
    CONFIG_NAME = './config/config_swingup.yml'
    meta_rollout_list = [0, 1, 2, 3]  # meta task id may not be the same as real task id
elif ENVS == 'Intersection':
    # intersection envs is difficult to merge into this file, it has a lot different functions
    sys.path.append('./envs/highway-env')
    import highway_env
    from mpc.mpc_is import MPC
    from train_on_highway import state_preprocess_full as state_preprocess
    from train_on_highway import gen_model_input_target_full as gen_model_pair
    CONFIG_NAME = './config/config_intersection.yml'
    meta_rollout_list = [0, 1, 2]
elif ENVS == 'Halfcheetah':
    sys.path.append('./envs/halfcheetah-env')
    import halfcheetah_envs
    from mpc.mpc_hc import MPC
    CONFIG_NAME = './config/config_halfcheetah.yml'
    meta_rollout_list = [0, 1]


def prepare_dynamics(gym_config):
    dynamics_name = gym_config['dynamics_name']
    #seed = gym_config['seed']
    dynamics_set = []
    for i in range(len(dynamics_name)):
        #env = gym.make(dynamics_name[i])
        # env.seed(seed)
        dynamics_set.append(gym.make(dynamics_name[i]))
    
    # use pre-defined env sequence
    task = [dynamics_set[i] for i in gym_config['task_dynamics_list']]
    return task


def load_config(config_path="config.yml"):
    if os.path.isfile(config_path):
        f = open(config_path)
        return yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise Exception("Configuration file is not found in the path: "+config_path)


if __name__ == '__main__':
    # dynamic model configuration
    config = load_config(CONFIG_NAME) 
    dpnn_config = config['DPNN_config']
    mpc_config = config['mpc_config']
    gym_config = config['gym_config']
    render = gym_config['render']
    nn_config = dpnn_config['NN_config']
    dp_config = dpnn_config["DP_config"]
    logger.info('Using environment: {}', ENVS)

    # initial MPC controller
    mpc_controller = MPC(mpc_config=mpc_config)

    # prepare task
    task = prepare_dynamics(gym_config)

    label_list = []
    assignment_predict_list = []
    task_solved = False
    total_count = 0
    task_epi = 0
    log = []
    log_name = datetime.datetime.now()

    # pretrain NN model if we dont want to load existing model
    pretrain = nn_config['model_config']['pretrain']
    if pretrain is True:
        logger.info('Pretrain a meta model')
        nn_model = NNComponent(nn_config)
        pretrain_episodes = 10
        for task_idx in range(len(task)):
            env = task[task_idx]
            # data collection
            for epi in range(pretrain_episodes):
                state = env.reset()
                if ENVS == 'Intersection':
                    state = state_preprocess(state)
                done = False
                if ENVS == 'Intersection':
                    # reset the controller at the beginning of each new dynamic
                    x_range = task[task_idx].config["observation"]["features_range"]["x"][1]
                    normalized_goal = task[task_idx].goal_pos/x_range
                    goal = np.concatenate((normalized_goal, task[task_idx].goal_heading), axis=0)
                    mpc_controller.reset(goal=goal, range=x_range)
                else:
                    mpc_controller.reset()
                while not done:
                    action = env.action_space.sample()
                    state_next, reward, done, info = env.step(action)
                    if ENVS == 'Intersection':
                        state_next = state_preprocess(state_next)
                        model_input, model_output = gen_model_pair(state, state_next)
                        nn_model.add_data_point([task_idx, model_input, action, model_output])
                    else:
                        nn_model.add_data_point([task_idx, state, action, state_next-state])
                    state = copy.deepcopy(state_next)
        nn_model.fit()
        nn_model.save_model(nn_model.save_model_path)

    # initialize the mixture model
    logger.info('Start to load meta model')
    model = DPNNMM(dp_config, nn_config)
    logger.info('Using model: {}', model.name)
    while task_epi < gym_config['task_episode']:
        task_epi += 1
        if total_count == 0:
            # for the first step, add one data pair with random policy as initialization
            state = task[0].reset()
            action = task[0].action_space.sample()
            state_next, reward, done, info = task[0].step(action)

            if ENVS == 'Intersection':
                state = state_preprocess(state)
                state_next = state_preprocess(state_next)
                model_input, model_output = gen_model_pair(state, state_next)
                model.add_data_point([0, model_input, action, model_output])
            else:
                model.add_data_point([0, state, action, state_next-state])
            model.fit()

            label_list.append(0)
            assignment_predict_list.append(model.DP_mix.assigns)

        # for other steps, run DPGP MBRL
        # Different sub-tasks share the same action space
        # Note that the subtask_index is unknown to the model, it's for debugging
        for subtask_index in range(len(task)):
            for epi in range(gym_config['subtask_episode']): # each subtask contains a fixed number of episode
                O, A, R, acc_reward, done = [], [], [], 0, False

                print('subtask: ', subtask_index, ', epi: ', epi)
                time_subtask_0 = time.time()
                state = task[subtask_index].reset()

                if ENVS == 'Intersection':
                    state = state_preprocess(state)
                    # reset the controller at the beginning of each new dynamic
                    x_range = task[subtask_index].config["observation"]["features_range"]["x"][1]
                    normalized_goal = task[subtask_index].goal_pos/x_range
                    goal = np.concatenate((normalized_goal, task[subtask_index].goal_heading), axis=0)
                    mpc_controller.reset(goal=goal, range=x_range)
                else:
                    mpc_controller.reset()

                while not done:
                    total_count += 1
                    if render:
                        task[subtask_index].render()

                    # MPC policy
                    if ENVS == 'Intersection':
                        # check single_vehicle
                        cur_state = state[:, 1:].reshape(1,-1).squeeze()
                        # if single vehicle, no collision check
                        single_vehicle = False
                        if ((cur_state[int(cur_state.shape[0]/2):] == 0)*1).mean() == 1:
                            single_vehicle = True
                        action = mpc_controller.act(task=task[subtask_index], model=model, state=cur_state, single_vehicle=single_vehicle)
                    else:
                        action = np.array([mpc_controller.act(task=task[subtask_index], model=model, state=state)])

                    # interact with env
                    state_next, reward, done, info = task[subtask_index].step(action)
                    acc_reward += reward
                    
                    if ENVS == 'Intersection':
                        state_next = state_preprocess(state_next)
                        model_input, model_output = gen_model_pair(state, state_next)
                    elif ENVS == 'Halfcheetah':
                        action = action[0]

                    A.append(action)
                    O.append(state_next)
                    R.append(reward)

                    # add the data
                    if ENVS == 'Intersection':
                        model.add_data_point([subtask_index, model_input, action, model_output])
                    else:
                        model.add_data_point([subtask_index, state, action, state_next-state])

                    # train the model
                    if model.stm_is_full:
                        model.fit()
                        for i in range(model.stm_length):
                            assignment_predict_list.append(model.DP_mix.assigns)
                            label_list.append(subtask_index)

                    state = copy.deepcopy(state_next)

                    if done:
                        samples = {
                            "obs": np.array(O),
                            "actions": np.array(A),
                            "rewards": np.array(R),
                            "reward_sum": acc_reward,
                        }
                        log.append(samples)
                        path = './misc/log/' + ENVS + '-' + model.name + '-' + log_name.strftime("%d-%H-%M") + '.npy'
                        np.save(path, log, allow_pickle=True)
                        dumb_reward_plot(path, PREFIX=ENVS + '-' + model.name + '-' + log_name.strftime("%d-%H-%M"))

                        path_assign = './misc/log/' + ENVS + '-' + log_name.strftime("%d-%H-%M") + '-assignment.npy'
                        assignment_result = {'predict': assignment_predict_list, 'label': label_list}
                        np.save(path_assign, [assignment_result], allow_pickle=True)
                        dumb_assignment_predict(path_assign, PREFIX=ENVS + '-' + model.name + '-' + log_name.strftime("%d-%H-%M"))

                        print('-------------------------------------------------')
                        print('Episode finished, time: ', time.time()-time_subtask_0, ' with acc_reward: ', acc_reward, ' with final reward: ', reward)
                        print('-------------------------------------------------')
