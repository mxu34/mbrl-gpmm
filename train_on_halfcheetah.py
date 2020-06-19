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
sys.path.append('./envs/halfcheetah-env')
import halfcheetah_envs

from utils import plot_index, dumb_reward_plot, save_test_succ_rate, Noise, dumb_predict_error_plot, dumb_assignment_predict
# use fetch slide mpccontroller
from mpc.mpc_hc import MPC

# all models
from dpgpmm.DPGPMM import DPGPMM
from baselines.SingleGP import SingleGP
from baselines.SingleSparseGP import SingleSparseGP
from baselines.NN import NN
from baselines.SNN import SNN
from dpgpmm.SDPGPMM import SDPGPMM
from baselines.NP import NP


def prepare_dynamics(gym_config):
    dynamics_name = gym_config['dynamics_name']
    #seed = gym_config['seed']
    dynamics_set = []
    for i in range(len(dynamics_name)):
        # env = gym.make(dynamics_name[i])
        # env.seed(seed)
        dynamics_set.append(gym.make(dynamics_name[i]))
    
    # use pre-defined env sequence
    task = [dynamics_set[i] for i in gym_config['task_dynamics_list']]
    return task


def halfcheetah_state_process(action, state, state_next):
    _action = action
    _state = state[0:]
    _state_next = state_next[0:]

    #_state = np.concatenate((state[0:1], state[3:10], state[12:18]))
    #_state_next = np.concatenate((state_next[0:1], state_next[3:10], state_next[12:18]))

    return _action, _state, _state_next


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
    config = load_config('./config/config_halfcheetah.yml')
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
    action_noise = Noise(mu=np.zeros(mpc_config['Random']['action_dim']), sigma=0.01)

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
    acc_test_reward_list = []
    log_name = datetime.datetime.now()
    test_iter = 10000
    test_length = 1
    pretrain_episodes = 15

    """ Meta Learning Stage """
    if model.name in ['NN', 'NP']:
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
                    model.data_process([0, obs, action, obs_next-obs])
                    obs = copy.deepcopy(obs_next)
        model.fit()
    
    while (not task_solved) and (task_epi < gym_config['task_episode']):
        task_epi += 1
        time_task_0 = time.time()
        if total_count == 0:
            # for the first step, add one data pair with random policy as initialization
            state = task[0].reset()
            action = task[0].action_space.sample()
            state_next, reward, done, info = task[0].step(action)
            # process data (parse state and delete one dimension of action)
            _action, _state, _state_next = halfcheetah_state_process(action, state, state_next)
            # train the model
            if model.name in ['NN', 'SNN', 'NP']:
                model.data_process(data=[0, _state, _action, _state_next - _state])
            elif model.name in ['SingleGP', 'SingleSparseGP']:
                model.add_point(x=[0, _state, _action, _state_next - _state])
            else:
                model.fit(data=[0, _state, _action, _state_next - _state])
            label_list.append(0)

        # for other steps, run DPGP MBRL
        # Different sub-tasks share the same action space
        # Note that the subtask_index is unknown to the model, it's for debugging
        task_r = 0
        for subtask_index in range(len(task)):
            for epi in range(gym_config['subtask_episode']): # each subtask contains a fixed number of episode
                '''
                # test stage
                if (epi+1) % test_iter == 0:
                    succ_rate = 0
                    for t_i in range(test_length):
                        state = task[subtask_index].reset()
                        mpc_controller.reset()
                        done = False
                        print('Testing: ', t_i)
                        acc_test_reward = 0
                        while not done:
                            if render:
                                task[subtask_index].render()
                            action = mpc_controller.act(task=task[subtask_index], model=model, state=state)
                            state_next, reward, done, info = task[subtask_index].step(action)
                            acc_test_reward += reward
                    acc_test_reward_list.append(acc_test_reward)
                    print('-------------------------------------------------')
                    print('Test success rate: ', acc_test_reward)
                    print('-------------------------------------------------')
                    save_test_succ_rate(acc_test_reward_list, PREFIX='HalfCheetah')
                '''
                # train stage
                O, A, R, acc_reward, done, E = [], [], [], 0, False, []
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
                    if epi == 0 and model.name in ['NN']:
                        action = task[subtask_index].action_space.sample()
                    else:
                        action = mpc_controller.act(task=task[subtask_index], model=model, state=state)# + action_noise.get_noise()

                    # interact with env
                    state_next, reward, done, info = task[subtask_index].step(action)
                    acc_reward += reward
                    #acc_reward = reward if reward > acc_reward else acc_reward

                    A.append(action)
                    O.append(state_next)
                    R.append(reward)

                    # process data (parse state and delete one dimension of action)
                    _action, _state, _state_next = halfcheetah_state_process(action, state, state_next)

                    ###################
                    ground_truth = _state_next-_state
                    if model.name in ['NN', 'SingleGP', 'NP']:
                        _state_pre, mse_error = model.test(_state[None], _action[None], ground_truth)
                    else:
                        _state_pre, mse_error = 0, 0

                    E.append(mse_error)


                    # train the model
                    if model.name in ['NN', 'SNN', 'NP'] and not done:
                        # Except DPGP, train the model at the end of one episode (even not every episode)
                        model.data_process(data=[subtask_index, _state, _action, _state_next-_state])
                    elif model.name in ['SingleGP',  'SingleSparseGP'] and not done:
                        model.add_point(x=[subtask_index, _state, _action, _state_next - _state])
                    else:
                        model.fit(data=[subtask_index, _state, _action, _state_next-_state])

                    # update state
                    state = copy.deepcopy(state_next)

                    if done:
                        #mse_error_train, mse_error_train_std = model.test_on_train_data()
                        mse_error_train = 0
                        mse_error_train_std = 0
                        samples = {
                            "obs": np.array(O),
                            "actions": np.array(A),
                            "rewards": np.array(R),
                            "reward_sum": acc_reward,
                            "mse_error": np.mean(E),
                            "mse_error_std": np.std(E),
                            "mse_error_train": mse_error_train,
                            "mse_error_train_std": mse_error_train_std,
                            "runtime": datetime.datetime.now().strftime("%d-%H-%M")
                        }
                        log.append(samples)
                        path = './misc/log/' + 'HalfCheetah-' + model.name + '-' + log_name.strftime("%d-%H-%M") + '.npy'
                        np.save(path, log, allow_pickle=True)

                        if model.name in ['DPGPMM']:
                            assignment_predict_list = []
                            for i in range(len(model.DP_mix.data)):
                                assignment_predict_list.append(model.DP_mix.assigns[i])
                            path_assign = './misc/log/' + log_name.strftime("%d-%H-%M") + '-assignment.npy'
                            assignment_result = {'predict': assignment_predict_list, 'label': label_list}
                            np.save(path_assign, [assignment_result], allow_pickle=True)
                            dumb_assignment_predict(path_assign, PREFIX='HalfCheetah'+'-'+log_name.strftime("%d-%H-%M"))

                        # draw some plots
                        dumb_reward_plot(path, PREFIX='HalfCheetah-'+model.name+'-'+ log_name.strftime("%d-%H-%M"),
                                         xlim=[0, 200], ylim=[-150, 100], y_line1=100, y_line2=100)
                        dumb_predict_error_plot(path, PREFIX='HalfCheetah'+'-'+log_name.strftime("%d-%H-%M"), ylim=[0, 5])

                        subtask_list.append(subtask_index)
                        subtask_reward.append(acc_reward)

                        if acc_reward > 100:
                            subtask_solved[subtask_index] = True
                            print('-------------------------------------------------')
                            print('Episode finished: Success!!!!, time: ', time.time()-time_subtask_0)
                            print('-------------------------------------------------')
                            subtask_list.append(subtask_index)
                            subtask_reward.append(acc_reward)
                            task_r += acc_reward

        task_reward.append(task_r)
        time_task = time.time() - time_task_0
        if np.sum(subtask_solved*1) == len(subtask_solved):
            logger.info('Solve all subtasks!')

        if task_epi % 1 == 0:
            print('***************************')
            print('task_episode: ', task_epi, ' time: ', time_task)
            if model.name in ['DPGPMM', 'SDPGPMM']:
                numbers = []
                for comp in model.DP_mix.comps:
                    numbers.append(comp.n)
                print('data in each component: ', numbers)
                with open('./misc/data_HalfCheetah.txt', 'a') as f:
                    f.write(str(task_epi) + ' ' + str(time_task) + ' ' + str(numbers) + ' ' + str(task_r) + '\n')
            else:
                with open('./misc/data_HalfCheetah_' + model.name + '.txt', 'a') as f:
                    f.write(str(task_epi) + ' ' + str(time_task) + ' ' + str(task_r) + '\n')
            print('***************************')
