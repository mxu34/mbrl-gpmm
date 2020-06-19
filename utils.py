

from loguru import logger
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


class Noise:
    """
    Based on the OpenAI baselines implementation, available at
    https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
    """
    def __init__(self, mu, sigma=0.1, theta=0.15, dt=0.05, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.x_prev = np.zeros_like(self.mu)

    def get_noise(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(
            size=self.mu.shape)
        self.x_prev = x
        return x
        
        
def print_three_models(assigns):
    print_list_1 = []
    print_list_2 = []
    print_list_3 = []
    for a_i in range(len(assigns.keys())):
        if a_i in assigns.keys():
            if a_i < 20:
                print_list_1.append(assigns[a_i])
            elif a_i >= 20 and a_i < 40:
                print_list_2.append(assigns[a_i])
            elif a_i >= 40 and a_i < 60:
                print_list_3.append(assigns[a_i])
            
    n_comps = len(np.unique(print_list_1+print_list_2+print_list_3))
    logger.info("number of cluster: {}", n_comps)
    logger.info("DP assignments:")

    if len(print_list_1) > 0:
        print('model 0: ', print_list_1)
    if len(print_list_2) > 0:
        print('model 1: ', print_list_2)
    if len(print_list_3) > 0:
        print('model 2: ', print_list_3)
    return n_comps


def plot_reward(subtask_list, rewards, name='Subtask Reward', xlabel='subtask index', y=196, scatter=True):
    plt.title(name)
    plt.axhline(y=y)
    if scatter:
        plt.plot(subtask_list, rewards, 'r.', markersize=7)
    else:
        plt.plot(subtask_list, rewards, 'r', linewidth=2)
    plt.ylabel('reward')
    plt.xlabel(xlabel)
    plt.savefig('./misc/' + name)
    # plt.show()
    plt.close()


def plot_index(predict_list, label_list, name='Cluster_Results.png'):
    plt.figure(figsize=(9,5))
    plt.subplot(211)
    plt.title(name)
    plt.plot(range(len(predict_list)), predict_list, 'r.', markersize=10, label='predict')
    plt.ylabel('Component Index')
    plt.legend()
    plt.subplot(212)
    plt.plot(range(len(predict_list)), label_list, 'g.', markersize=10, label='groundtruth')
    plt.xlabel('Data Index')
    plt.ylabel('Component Index')
    plt.legend()
    plt.savefig('./misc/' + name)
    #plt.show()
    plt.close()

def plot_index_results(data, predict_list, label_list):
    plt.figure(figsize=(9,5))
    plt.subplot(211)
    plt.title('Cluster Results with GPytorch')
    plt.plot(range(data.shape[0]), predict_list, 'r.', markersize=10, label='predict')
    plt.ylabel('Component Index')
    plt.legend()
    plt.subplot(212)
    plt.plot(range(data.shape[0]), label_list, 'g.', markersize=10, label='groundtruth')
    plt.xlabel('Data Index')
    plt.ylabel('Component Index')
    plt.legend()
    plt.savefig('./misc/Cluster Results with GPytorch')
    plt.show()


def plot_gibbs_time(data, time_record):
    plt.figure()
    plt.plot(range(data.shape[0]), time_record)
    plt.xlabel('Data Index')
    plt.ylabel('Time per Datapoint (s)')
    plt.title('Cost time with GPytorch')
    plt.savefig('./misc/Cost time with GPytorch')
    plt.show()

def log_plot(path):
    data = np.load(path)

def check_create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def dumb_assignment_predict(path, PREFIX=''):
    assignment_result = np.load(path, allow_pickle=True)[0]
    predict = assignment_result['predict']
    label = assignment_result['label']
    name = PREFIX + '_assignment_predict.png'
    plot_index(predict, label, name)

def dumb_predict_error_plot(path, PREFIX='', ylim=[0, 500]):
    data_list = np.load(path, allow_pickle=True)

    err_list = []
    err_list_std = []
    err_train = []
    err_train_std = []
    for data in data_list:
        err_list.append(data["mse_error"])
        err_list_std.append(data["mse_error_std"])
        err_train.append(data["mse_error_train"])
        err_train_std.append(data["mse_error_train_std"])
    
    name = PREFIX + '_predict_error'
    plt.figure(figsize=(6,4))
    plt.title(name)
    plt.plot(range(len(err_list)), err_list, linewidth=1, color='royalblue', label='Test Error')
    lower = [a-b for a, b in zip(err_list, err_list_std)]
    upper = [a+b for a, b in zip(err_list, err_list_std)]
    plt.fill_between(range(len(err_list)), lower, upper, facecolor='royalblue', alpha=0.3)

    plt.plot(range(len(err_train)), err_train, linewidth=1, color='deeppink', label='Train Error')
    lower = [a-b for a, b in zip(err_train, err_train_std)]
    upper = [a+b for a, b in zip(err_train, err_train_std)]
    plt.fill_between(range(len(err_train)), lower, upper, facecolor='deeppink', alpha=0.3)

    plt.legend()
    plt.grid()
    plt.ylabel('error')
    plt.xlabel('timestep')
    plt.ylim(ylim)
    plt.savefig('./misc/' + name)
    plt.close('all')

def dumb_reward_plot(path, PREFIX='', xlim=[0, 200], ylim=[0, 200], y_line1=1, y_line2=175):
    # path = './misc/log/April-14-19-55.npy'
    data_list = np.load(path, allow_pickle=True)
    name = PREFIX + '_task_reward'

    plt.figure(figsize=(6,4))
    plt.title(name)
    if y_line1:
        plt.axhline(y=y_line1, color='k')

    acc_r_list = []
    for data in data_list:
        obs = data["obs"]
        rewards = data["rewards"]
        acc_r = data["reward_sum"]
        acc_r_list.append(acc_r)
        num_step = len(rewards)
        plt.plot(range(rewards.shape[0]), rewards, linewidth=1, label=str(round(acc_r, 4)))

    plt.ylabel('reward')
    plt.xlabel('timestep')
    plt.xlim(xlim)
    plt.ylim([-5, 5])
    # plt.legend()
    plt.savefig('./misc/' + name)
    plt.close('all')
    # plt.show()

    plt.figure(figsize=(6, 4))
    name = PREFIX + '_acc_task_reward'
    plt.title(name)
    if y_line2:
        plt.axhline(y=y_line2)
    plt.plot(range(len(acc_r_list)), acc_r_list, 'C2', linewidth=2)
    plt.grid()

    '''
    color_bar = ['green', 'yellow', 'red', 'blue']
    p_1 = [0,  1,  2, 12, 13, 14, 24, 25, 26, 36, 37, 38]
    p_2 = [3,  4,  5, 15, 16, 17, 27, 28, 29, 39, 40, 41]
    p_3 = [6,  7,  8, 18, 19, 20, 30, 31, 32, 42, 43, 44]
    p_4 = [9, 10, 11, 21, 22, 23, 33, 34, 35, 45, 46, 47]
    for p_i in range(1, len(data_list)):
        if p_i in p_1:
            c_i = color_bar[0]
        elif p_i in p_2:
            c_i = color_bar[1]
        elif p_i in p_3:
            c_i = color_bar[2]
        else:
            c_i = color_bar[3]
        plt.fill_between([p_i-1, p_i], ylim[0], ylim[1], facecolor=c_i, alpha=0.2)
    '''
    
    plt.ylim(ylim)
    plt.ylabel('acc_reward')
    plt.xlabel('episode')
    # plt.legend()
    plt.savefig('./misc/' + name)
    plt.close('all')
    # plt.show()


def save_test_succ_rate(succ_rate_list, PREFIX=''):
    plt.figure(figsize=(6, 4))
    name = PREFIX + '_test_acc_reward'
    plt.title(name)
    plt.axhline(y=1.0)
    plt.plot(range(len(succ_rate_list)), succ_rate_list, 'C2', linewidth=2)
    plt.grid()
    plt.savefig('./misc/' + name)
    plt.close('all')
