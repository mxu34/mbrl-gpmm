import numpy as np
from tqdm import trange, tqdm
from .optimizers import RandomOptimizer, CEMOptimizer
import copy
import sys
sys.path.append('../envs/highway-env')
from highway_env.utils import rotated_rectangles_intersect

class MPC(object):
    optimizers = {"CEM": CEMOptimizer, "Random": RandomOptimizer}

    def __init__(self, mpc_config):
        # mpc_config = config["mpc_config"]
        self.type = mpc_config["optimizer"]
        conf = mpc_config[self.type]
        self.horizon = conf["horizon"]
        self.gamma = conf["gamma"]
        self.action_low = np.array(conf["action_low"]) # array (dim,)
        self.action_high = np.array(conf["action_high"]) # array (dim,)
        self.action_dim = conf["action_dim"]
        self.popsize = conf["popsize"]
        self.action_cost = conf["action_cost"]
        self.particle = conf["particle"]

        self.init_mean = np.array([conf["init_mean"]] * self.horizon)
        self.init_var = np.array([conf["init_var"]] * self.horizon)

        if len(self.action_low) == 1: # auto fill in other dims
            self.action_low = np.tile(self.action_low, [self.action_dim])
            self.action_high = np.tile(self.action_high, [self.action_dim])
        
        self.optimizer = MPC.optimizers[self.type](sol_dim=self.horizon*self.action_dim,
                                                   popsize=self.popsize,
                                                   upper_bound=np.array(conf["action_high"]),
                                                   lower_bound=np.array(conf["action_low"]),
                                                   max_iters=conf["max_iters"],
                                                   num_elites=conf["num_elites"],
                                                   epsilon=conf["epsilon"],
                                                   alpha=conf["alpha"])

        self.optimizer.setup(self.intersection_cost_function)

    def reset(self, goal, range):
        """Resets this controller (clears previous solution, calls all update functions).

        Returns: None
        """
        # print('set init mean to 0')
        self.goal_pos = goal[0:2]
        self.goal_heading = goal[2]
        self.range = range
        self.prev_sol = np.tile((self.action_low + self.action_high) / 2, [self.horizon])
        self.init_var = np.tile(np.square(self.action_low - self.action_high) / 16, [self.horizon])

    def act(self, task, model, state,
            single_vehicle = False,
            ground_truth=False):
        '''
        :param state: task, model, (numpy array) current state
        :return: (float) optimal action
        '''

        self.task = task
        self.model = model
        self.state = state
        self.single_vehicle = single_vehicle
        self.ground_truth = ground_truth

        soln, var = self.optimizer.obtain_solution(self.prev_sol, self.init_var)
        self.prev_sol = np.concatenate([np.copy(soln)[self.action_dim:], np.zeros(self.action_dim)])
        action = soln[:self.action_dim]
        # print('soln', self.prev_sol.reshape((-1, self.horizon, self.action_dim)))
        return action

    def intersection_cost_function(self, actions):
        """
        Calculate the cost given a sequence of actions
        Parameters:
        ----------
            @param numpy array - actions : size should be (batch_size x horizon number)

        Return:
        ----------
            @param numpy array - cost : length should be of batch_size
        """

        actions = actions.reshape((-1, self.horizon, self.action_dim)) # [pop size, horizon, action_dim]
        actions = np.tile(actions, (self.particle, 1, 1))

        costs = np.zeros(self.popsize*self.particle)
        state = np.repeat(self.state.reshape(1, -1), self.popsize*self.particle, axis=0)

        crash = np.zeros_like(costs)

        for t in range(self.horizon):
            action = actions[:, t, :]  # numpy array (batch_size x action dim)
            # the output of the prediction model is [state_next - state]
            state_next = self.model.predict(state, action) + state
            cost, crash_sign = self.intersection_cost(state_next, action, t)  # compute cost
            crash += crash_sign
            # cost[crash >= 1] = -10

            costs += cost * self.gamma**t
            state = copy.deepcopy(state_next)

        # average between particles
        costs = np.mean(costs.reshape((self.particle, -1)), axis=0)
        return costs

    def preprocess(self, state):
        pos = state[:, 0:2]
        vel = state[:, 2:4]
        cos_h = state[:, 4]
        sin_h = state[:, 5]
        return pos, vel, cos_h, sin_h

    def single_check_collision(self, ego_pos, ego_heading, other_pos, other_heading):
        LENGTH = 5.0/self.range
        WIDTH = 2.0/self.range
        ego_crashed = 0

        if np.linalg.norm(other_pos - ego_pos) <= LENGTH:
            if rotated_rectangles_intersect((ego_pos, LENGTH, WIDTH, ego_heading),
                                              (other_pos, LENGTH, WIDTH, other_heading)):
                ego_crashed = 1

        return ego_crashed

    def check_collision(self, ego_info, other_info):

        # other_sin_h[other_sin_h>1] = 1
        # other_sin_h[other_sin_h<-1] = -1
        # other_cos_h[other_cos_h>1] = 1
        # other_cos_h[other_cos_h<-1] = -1

        ego_pos = ego_info[:, 0:2]
        ego_sin_h = ego_info[:, 5]
        ego_cos_h = ego_info[:, 4]

        other_pos = other_info[:, 0:2]
        other_sin_h = other_info[:, 5]
        other_cos_h = other_info[:, 4]

        if other_cos_h.any() > 1 or other_cos_h.any() < -1 or other_sin_h.any() > 1 or other_sin_h.any() < -1:
            print('!!!!!!!!!!!!sin and cos out of range')

        ego_heading = np.arctan2(ego_sin_h, ego_cos_h)
        other_heading = np.arctan2(other_sin_h, other_cos_h)

        ego_crashed = np.zeros(ego_pos.shape[0])
        for i in range(ego_pos.shape[0]):
            ego_crashed[i] = self.single_check_collision(ego_pos[i], ego_heading[i],
                                                         other_pos[i], other_heading[i])
        return ego_crashed

    def intersection_cost(self, state, action, t = None):

        # prepare state, change relative coordinate to absolute coordinate
        state_dim = int(state.shape[1] / 2)
        ego_info = state[:, 0:state_dim]
        other_info = state[:, 0:state_dim]+state[:, state_dim:]

        ego_pos, ego_vel, ego_cos_h, ego_sin_h = self.preprocess(ego_info)
        other_pos = other_info[:, 0:2]
        ego_velocity = np.sqrt(np.sum(ego_vel**2, axis=1))
        ego_other_distance = np.sqrt(np.sum((other_pos - ego_pos) ** 2, axis=1))

        # check crashed
        ego_crashed = 0
        if not self.single_vehicle:
            ego_crashed = self.check_collision(ego_info, other_info)
            # if ego_crashed.any() == 1:
                # print('horizion: ', t, ' crash detected, percentage', np.mean(ego_crashed))

        # calculate reward
        COLLISION_REWARD = -5
        HIGH_VELOCITY_REWARD = 0.1
        POS_ARRIVED_REWARD_x = 1
        POS_ARRIVED_REWARD_y = 5
        HEADING_ARRIVED_REWARD = 1
        ACTION_REWARD = 0.05
        DISTANCE_REWARD = 2
        OUT_OF_ROAD_REWARD = -5

        reward = COLLISION_REWARD * ego_crashed
        x_distance = np.sqrt((self.goal_pos[0] - ego_pos[:, 0])**2)
        y_distance = np.sqrt((self.goal_pos[1] - ego_pos[:, 1])**2)
        heading_distance = np.abs(self.goal_heading - ego_cos_h)

        r_vel = HIGH_VELOCITY_REWARD * ego_velocity
        r_x_pos = POS_ARRIVED_REWARD_x * np.exp(-x_distance)
        r_y_pos = POS_ARRIVED_REWARD_y * np.exp(-y_distance*5)
        r_heading = HEADING_ARRIVED_REWARD * np.exp(-heading_distance)
        reward_action = - ACTION_REWARD * action[:, 1] ** 2
        reward_distance = - DISTANCE_REWARD * np.exp(- ego_other_distance)

        # reward += r_x_pos
        reward += r_y_pos
        reward += r_heading
        reward += r_vel
        reward += reward_action
        reward += reward_distance
        # reward_heading_pos = np.exp(-(pos_distance ** 2 + heading_distance ** 2))
        cost = -reward
        return cost, ego_crashed
