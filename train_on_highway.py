import time, datetime
import copy
import os
import sys
import warnings
#warnings.filterwarnings("ignore", category=UserWarning)
#warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
from loguru import logger
import yaml
import argparse

import gym
sys.path.append('./envs/highway-env')
import highway_env

from utils import plot_reward, plot_index, dumb_reward_plot, dumb_assignment_predict
from mpc.mpc_is import MPC

# all models
from dpgpmm.DPGPMM import DPGPMM
from baselines.SingleGP import SingleGP
# from baselines.SingleSparseGP import SingleSparseGP
from baselines.NN import NN
from baselines.NP import NP


def prepare_dynamics(gym_config):
    dynamics_name = gym_config['dynamics_name']
    seed = gym_config['seed']
    dynamics_set = []
    for i in range(len(dynamics_name)):
        env = gym.make(dynamics_name[i])
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


def state_preprocess_full(s):
    # Input state: "presence", "id", "x", "y", "vx", "vy", "cos_h", "sin_h"
    # output features: "id", "x", "y", "vx", "vy", "cos_h", "sin_h"
    # Output shape: [num_vehicles, num_features]
    index = s[:,0] == 1
    s = s[index, 1:]
    if s.shape[0] == 1:
        s_common = np.zeros_like(s)
        s = np.concatenate((s, s_common), axis=0)
    return s


def gen_model_input_target_full(s, s_n, absolute=False):
    # all the close vehicles
    s_other = s[1:, :]
    s_n_other = s_n[1:, :]

    # if no other vehicles
    if s_other.shape[0] == 0 or s_other.shape[0] == 0:
        s_common = np.zeros((1, s_other.shape[1]-1))
        s_n_common = np.zeros((1, s_other.shape[1]-1))
    else:
        id_s = s_other[:, 0]
        id_s_n = s_other[:, 0]
        intersection, s_index, s_n_index = np.intersect1d(id_s, id_s_n, return_indices=True)

        # if no common vehicle
        if intersection.shape[0] == 0:
            s_common = np.zeros((1, s_other.shape[1]-1))
            s_n_common = np.zeros((1, s_other.shape[1]-1))
        else:
            s_common = s_other[s_index, 1:]
            s_n_common = s_n_other[s_n_index, 1:]

    # add ego vehicle
    model_input = np.concatenate((s[0:1, 1:], s_common), axis=0).reshape((1,-1))
    model_output = np.concatenate((s_n[0:1, 1:], s_n_common), axis=0).reshape((1,-1)) - model_input

    # return input, output

    if model_output.shape[1] > 2*(s.shape[1]-1) or model_input.shape[1] > 2*(s.shape[1]-1):
        raise ValueError('Why there are two other vehicles')
    return model_input.squeeze(), model_output.squeeze()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process model selection.')
    parser.add_argument('--model', type=str, default='GPMM',
                        help='model name required')
    args = parser.parse_args()

    # configuration
    config = load_config('./config/config_intersection.yml')
    dpgp_config = config['DPGP_config']
    representation = dpgp_config["mode"]
    dpgp_config = dpgp_config[representation]
    gp_config = config['SingleGP_config']
    sparse_gp_config = config['SingleSparseGP_config']
    nn_config = config['NN_config']
    np_config = config['NP_config']
    mpc_config = config['mpc_config']
    gym_config = config['gym_config']
    render = gym_config['render']

    if representation == "FullState":
        state_preprocess = state_preprocess_full
        gen_model_pair = gen_model_input_target_full

    # initialize the mixture model
    if args.model == 'GP':
        model = SingleGP(gp_config=gp_config)
    elif args.model == 'NP':
        model = NP(NP_config=np_config)
    elif args.model == 'NN':
        model = NN(NN_config=nn_config)
    elif args.model == 'GPMM':
        model = DPGPMM(dpgp_config=dpgp_config)

    logger.info('Using model: {}', model.name)

    # initial MPC controller
    mpc_controller = MPC(mpc_config=mpc_config)

    # prepare task
    # the task is solved, if each dynamic is solved
    task = prepare_dynamics(gym_config)

    # prepare goal: unprotected left turn
    absolute_goal = np.array([-10, -2])

    """start DPGP-MBRL"""
    data_buffer = []
    label_list = []
    subtask_list = []
    subtask_reward = []
    subtask_succ_count = [0]
    comp_trainable = [1]
    task_reward = []
    trainable = True
    task_solved = False
    subtask_solved = [False, False, False, False]
    total_count = 0
    task_epi = 0
    log = []
    log_name = None

    """pretrain"""
    # if model.name == 'NN' or (model.name == 'NP' and args.pretrain== 1):
    if model.name == 'NN' or (model.name == 'NP'):
        pretrain_episodes = 15
        for task_idx in range(len(task)):
            env = task[task_idx]
            # data collection
            for epi in range(pretrain_episodes):
                state = env.reset()
                state = state_preprocess(state)
                done = False
                # reset the controller at the beginning of each new dynamic
                #x_range = env.config["observation"]["features_range"]["x"][1]
                #normalized_goal = env.goal_pos / x_range
                #goal = np.concatenate((normalized_goal, env.goal_heading), axis=0)
                #mpc_controller.reset(goal=goal, range=x_range)
                while not done:
                    action = env.action_space.sample()
                    state_next, reward, done, info = env.step(action)
                    state_next = state_preprocess(state_next)
                    model_input, model_output = gen_model_pair(state, state_next)
                    model.data_process([0, model_input, action, model_output])
                    state = copy.deepcopy(state_next)
        # training the model
        model.validation_flag = True
        # model.n_epochs = 20
        model.fit()

    """online trainingS"""
    while (not task_solved) and (task_epi < gym_config['task_episode']):
        task_epi += 1
        time_task_0 = time.time()
        if total_count == 0:
            # for the first step, add one data pair with random policy as initialization
            state = task[0].reset()
            action = task[0].action_space.sample()
            state_next, reward, done, info = task[0].step(action)

            # state and target preprocess
            state = state_preprocess(state)
            state_next = state_preprocess(state_next)
            model_input, model_output = gen_model_pair(state, state_next)
            model.fit(data=[0, model_input, action, model_output])

            label_list.append(0)

        # for other steps, run DPGP MBRL
        task_r = 0
        for subtask_index in range(len(task)):
            print('-------------------')
            print('subtask_index ', subtask_index)
            for epi in range(gym_config['subtask_episode']): # each subtask contains a fixed number of episode
                print('-------------------')
                print('epi ', epi)
                O, A, R, acc_reward, done = [], [], [], 0, False
                time_subtask_0 = time.time()
                print('subtask: ', subtask_index, ', epi: ', epi)

                state = task[subtask_index].reset()
                state = state_preprocess(state)
                O.append(state)

                # reset the controller at the beginning of each new dynamic
                x_range = task[subtask_index].config["observation"]["features_range"]["x"][1]
                normalized_goal = task[subtask_index].goal_pos/x_range
                goal = np.concatenate((normalized_goal, task[subtask_index].goal_heading), axis = 0)
                mpc_controller.reset(goal=goal, range = x_range)

                while not done:
                    if render:
                        task[subtask_index].render()
                    total_count += 1
                    label_list.append(subtask_index)

                    # MPC policy
                    cur_state = state[:, 1:].reshape(1,-1).squeeze()
                    # action = ["acceleration", "steering"]
                    # if single vehicle, no collision check
                    single_vehicle = False
                    if ((cur_state[int(cur_state.shape[0]/2):] == 0)*1).mean() == 1:
                        single_vehicle = True
                    print('*********')
                    # print('collision check: ', not single_vehicle)
                    start_1 = time.time()
                    action = mpc_controller.act(task=task[subtask_index], model=model, state=cur_state, single_vehicle=single_vehicle)
                    start_2 = time.time()

                    print('acceleration, steering ', action)

                    # Random Policy
                    # action = task[subtask_index].action_space.sample()

                    # interact with env
                    state_next, reward, done, info = task[subtask_index].step(action)
                    state_next = state_preprocess(state_next)
                    acc_reward += reward

                    A.append(action)
                    O.append(state_next)
                    R.append(reward)

                    start_3 = time.time()

                    # train the model
                    # when reach some kind of metric, stop training, only inference
                    model_input, model_output = gen_model_pair(state, state_next)

                    if model.name == 'DPGPMM':
                        model.fit(data=[subtask_index, model_input, action, model_output])
                    elif model.name == 'SingleGP':
                        model.fit(data=[subtask_index, model_input, action, model_output])
                    elif model.name == 'NN' or model.name == 'NP':
                        model.data_process([0, model_input, action, model_output])
                    else:
                        raise NotImplementedError

                    state = copy.deepcopy(state_next)
                    start_4 = time.time()

                    # print('mpc: {}, env: {}, model: {}'.format(start_2-start_1, start_3-start_2, start_4-start_3))

                    if done:
                        if model.name == 'NN' or model.name == 'NP':
                            model.fit()
                            
                        task[subtask_index].close()
                        samples = {
                            "obs": np.array(O),
                            "actions": np.array(A),
                            "rewards": np.array(R),
                            "reward_sum": acc_reward,
                        }
                        log.append(samples)
                        if log_name is None:
                            log_name = datetime.datetime.now()
                        env_name = gym_config["dynamics_name"][subtask_index]
                        path = './misc/log/' + 'Intersection-' + model.name + '-' + log_name.strftime("%d-%H-%M") + '.npy'
                        np.save(path, log, allow_pickle=True)
                        dumb_reward_plot(path, PREFIX='Intersection-'+model.name+'-'+ gym_config['dynamics_name'][0]+'-'+ log_name.strftime("%d-%H-%M"),
                                         xlim=[0, 40], ylim=[0, 80], y_line1=2, y_line2=50)

                        if model.name in ['DPGPMM']:
                            assignment_predict_list = []
                            for i in range(len(model.DP_mix.data)):
                                assignment_predict_list.append(model.DP_mix.assigns[i])
                            path_assign = './misc/log/' + log_name.strftime("%d-%H-%M") + '-assignment.npy'
                            assignment_result = {'predict': assignment_predict_list, 'label': label_list}
                            np.save(path_assign, [assignment_result], allow_pickle=True)
                            dumb_assignment_predict(path_assign, PREFIX='Intersection'+'-'+log_name.strftime("%d-%H-%M"))

                        print('-------------------------------------------------')
                        print('Episode finished, time: ', time.time()-time_subtask_0, ' with acc_reward: ', acc_reward, ' with final reward: ', reward)
                        print('-------------------------------------------------')
                        subtask_list.append(subtask_index)
                        subtask_reward.append(acc_reward)

                        task_r += acc_reward

        task_reward.append(task_r)
        time_task = time.time() - time_task_0

        if task_epi % 1 == 0:
            print('***************************')
            print('task_episode: ', task_epi, ' time: ', time_task)
            if model.name == 'DPGPMM':
                numbers = []
                for comp in model.DP_mix.comps:
                    numbers.append(comp.n)
                print('data in each component: ', numbers)
                print('***************************')
                with open('./misc/data_Intersection.txt', 'a') as f:
                    f.write(str(task_epi) + ' ' + str(time_task) + ' ' + str(numbers) + ' ' + str(
                        task_r) + '\n')
            else:
                with open('./misc/data_Intersection_' + model.name + '.txt', 'a') as f:
                    f.write(str(task_epi) + ' ' + str(time_task) + ' ' + str(task_r) + '\n')
                print('***************************')
