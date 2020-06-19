import time, datetime
import copy
import os
import sys
import argparse
import warnings
#warnings.filterwarnings("ignore", category=UserWarning)
#warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
from loguru import logger
import yaml

import gym
sys.path.append('./envs/cartpole-envs')
sys.path.append('./envs/highway-env')
import cartpole_envs

from utils import plot_reward, plot_index, dumb_reward_plot, dumb_assignment_predict
from mpc.mpc_cp import MPC

# all models
from dpgpmm.DPGPMM import DPGPMM
from baselines.SingleGP import SingleGP
from baselines.SingleSparseGP import SingleSparseGP
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process model selection.')
    parser.add_argument('--model', type=str, default='GPMM',
                        help='model name required')
    args = parser.parse_args()

    # dynamic model configuration
    config = load_config('./config/config_swingup.yml')
    dpgp_config = config['DPGP_config']
    gp_config = config['SingleGP_config']
    sparse_gp_config = config['SingleSparseGP_config']
    nn_config = config['NN_config']
    np_config = config['NP_config']
    mpc_config = config['mpc_config']
    gym_config = config['gym_config']
    render = gym_config['render']

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

    # pretrain NN model
    if model.name == 'NN' or (model.name == 'NP'):
        pretrain_episodes = 10
        print('pretrain~~~~~~~~~~~~~~~~~~~~~~~')
        for task_idx in range(len(task)):
            env = task[task_idx]
            # data collection
            for epi in range(pretrain_episodes):
                obs = env.reset()
                done = False
                mpc_controller.reset()
                while not done:
                    action = env.action_space.sample()
                    obs_next, reward, done, state_next = env.step(action)
                    model.data_process([0, obs, action, obs_next - obs])
                    obs = copy.deepcopy(obs_next)
        # training the model
        model.validation_flag = True
        # model.n_epochs = 20
        model.fit()

    while (not task_solved) and (task_epi < gym_config['task_episode']):
        task_epi += 1
        time_task_0 = time.time()
        if total_count == 0:
            # for the first step, add one data pair with random policy as initialization
            state = task[0].reset()
            action = task[0].action_space.sample()
            state_next, reward, done, info = task[0].step(action)
            # train the model
            if model.name in ['NN', 'SNN', 'NP']:
                model.data_process(data=[0, state, action, state_next - state])
            elif model.name in ['SingleGP', 'SingleSparseGP']:
                model.add_point(x=[0, state, action, state_next - state])
            else:
                model.fit(data=[0, state, action, state_next - state])

            label_list.append(0)

        # Different sub-tasks share the same action space
        # Note that the subtask_index is unknown to the model, it's for debugging
        task_r = 0
        for subtask_index in range(len(task)):
            for epi in range(gym_config['subtask_episode']): # each subtask contains a fixed number of episode
                O, A, R, acc_reward, done = [], [], [], 0, False

                print('subtask: ', subtask_index, ', epi: ', epi)
                time_subtask_0 = time.time()

                state = task[subtask_index].reset()
                O.append(state)
                # reset the controller at the beginning of each new dynamic
                mpc_controller.reset()

                while not done:
                    if render:
                        task[subtask_index].render()

                    total_count += 1
                    label_list.append(subtask_index)

                    # MPC policy
                    start_1 = time.time()
                    action = np.array([mpc_controller.act(task=task[subtask_index], model=model, state=state)])
                    start_2 = time.time()

                    # Random Policy
                    # action = task[subtask_index].action_space.sample()

                    # interact with env
                    state_next, reward, done, info = task[subtask_index].step(action)
                    acc_reward += reward

                    print('action ', action)
                    print('reward: %.4f' % reward)

                    A.append(action)
                    O.append(state_next)
                    R.append(reward)

                    # train the model
                    if model.name in ['NN', 'SNN', 'NP'] and not done:
                        # Except DPGP, train the model at the end of one episode (even not every episode)
                        model.data_process(data=[subtask_index, state, action, state_next-state])
                    else:
                        model.fit(data=[subtask_index, state, action, state_next-state])

                    state = copy.deepcopy(state_next)

                    if done:
                        samples = {
                            "obs": np.array(O),
                            "actions": np.array(A),
                            "rewards": np.array(R),
                            "reward_sum": acc_reward,
                        }
                        log.append(samples)
                        if log_name is None:
                            log_name = datetime.datetime.now()
                        path = './misc/log/CartPole-'+model.name +'-' + log_name.strftime("%d-%H-%M") + '.npy'
                        np.save(path, log, allow_pickle=True)
                        dumb_reward_plot(path, PREFIX='CartPole-'+model.name+'-'+ log_name.strftime("%d-%H-%M"))

                        if model.name in ['DPGPMM']:
                            assignment_predict_list = []
                            for i in range(len(model.DP_mix.data)):
                                assignment_predict_list.append(model.DP_mix.assigns[i])
                            path_assign = './misc/log/' + log_name.strftime("%d-%H-%M") + '-assignment.npy'
                            assignment_result = {'predict': assignment_predict_list, 'label': label_list}
                            np.save(path_assign, [assignment_result], allow_pickle=True)
                            dumb_assignment_predict(path_assign, PREFIX='Cartpole'+'-'+ log_name.strftime("%d-%H-%M"))

                        print('-------------------------------------------------')
                        print('Episode finished, time: ', time.time()-time_subtask_0, ' with acc_reward: ', acc_reward, ' with final reward: ', reward)
                        print('-------------------------------------------------')
                        subtask_list.append(subtask_index)
                        subtask_reward.append(acc_reward)
                        task_r += acc_reward
                if render:
                    task[subtask_index].close()

        task_reward.append(task_r)
        time_task = time.time() - time_task_0

        if task_epi % 1 == 0:
            # record the reward
            plot_reward(subtask_list, subtask_reward,
                        name=model.name + '_' + 'subtask_reward_CartPole_SwingUp_' + str(task_epi), xlabel='subtask', y=195)
            plot_reward(range(len(task_reward)), task_reward,
                        name=model.name + '_' + 'task_reward_CartPole_SwingUp_' + str(task_epi), xlabel='episode', y=1560, scatter=False)

            print('***************************')
            print('task_episode: ', task_epi, ' time: ', time_task)
            if model.name == 'DPGPMM':
                numbers = []
                for comp in model.DP_mix.comps:
                    numbers.append(comp.n)
                print('data in each component: ', numbers)
                print('***************************')
                with open('./misc/data_SwingUp.txt', 'a') as f:
                    f.write(str(task_epi) + ' ' + str(time_task) + ' ' + str(numbers) + ' ' + str(task_r) + '\n')
            else:
                with open('./misc/data_SwingUp_' + model.name + '.txt', 'a') as f:
                    f.write(str(task_epi) + ' ' + str(time_task) + ' ' + str(task_r) + '\n')
                print('***************************')
