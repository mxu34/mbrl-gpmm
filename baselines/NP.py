

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from loguru import logger
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from .ANP.NPModel import NeuralProcessModel
import time

def CUDA(var):
    return var.cuda() if torch.cuda.is_available() else var


class NP(object):
    # output: [state mean, state var]
    name = "NP"

    def __init__(self, NP_config, dataset=None):
        super().__init__()
        model_config = NP_config["model_config"]
        training_config = NP_config["training_config"]

        self.state_dim = model_config["state_dim"]
        self.action_dim = model_config["action_dim"]
        self.x_dim = self.state_dim + self.action_dim
        self.y_dim = model_config["state_dim"]
        self.strategy = model_config["strategy"]
        self.context_number = model_config["context_number"]
        self.target_number = model_config["target_number"]

        self.n_epochs = training_config["n_epochs"]
        self.lr = training_config["learning_rate"]
        self.batch_size = training_config["batch_size"]

        self.save_model_flag = training_config["save_model_flag"]
        self.save_model_path = training_config["save_model_path"]

        self.validation_flag = training_config["validation_flag"]
        self.validate_freq = training_config["validation_freq"]
        self.validation_ratio = training_config["validation_ratio"]

        # TODO: Let NP use the config file
        # NOTICE: NP configuration

        # NOTICE: 'gp' use all existing data as context, input data as target
        #           (do the inference conditioned on previous data)
        #         'nn' use input data as both context and target
        #           (NP learns the "function" rather than memorizing data)
        # print('model_config["likelihood_method"] =', model_config["likelihood_method"])
        self.likelihood_method = model_config["likelihood_method"]  # 'gp' or 'nn'
        if self.likelihood_method not in ['gp', 'nn']:
            print("Please select inference method, 'gp' or 'nn'!")

        # NOTICE: 'loss' use the negative loss to decide the 'likelihood' F(data, model)
        #          'll' use the sum of the likelihood of F(data, model) = N(data , pred_mu, pred_var)
        self.likelihood_value = model_config['likelihood_value']   # 'll' or 'loss'
        if self.likelihood_value not in ['ll', 'loss']:
            print("Please select likelihood value, 'loss' or 'll'!")

        self.sequential = model_config['sequential']
        self.virtual_batch = model_config['virtual_batch']
        self.np_hidden_list = model_config['np_hidden_list']
        self.np_latent_dim = model_config['np_latent_dim']

        if model_config["load_model"]:
            self.model = CUDA(torch.load(model_config["model_path"]))
        else:
            self.model = CUDA(NeuralProcessModel(x_dim=self.x_dim,
                                                 y_dim=self.y_dim,
                                                 mlp_hidden_size_list=self.np_hidden_list,
                                                 latent_dim=self.np_latent_dim,
                                                 use_rnn=False,
                                                 use_self_attention=True,
                                                 use_deter_path=True))

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.X = None
        self.Y = None

    # Reset the weight of the model, might be useful
    def reset_model(self):
        def weight_reset(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, 0.0, 0.02)
        self.model.apply(weight_reset)

    # NOTICE: data process
    #  from baselines.NN
    def data_process(self, data):
        s = data[1][None]
        a = data[2][None]
        label = data[3][None] # here label means the next state
        data = np.concatenate((s, a), axis=1)

        # add new data point to data buffer
        if self.X is None:
            self.X = CUDA(torch.Tensor(data))
            self.Y = CUDA(torch.Tensor(label))
        else:
            self.X = torch.cat((self.X, CUDA(torch.tensor(data).float())), dim=0)
            self.Y = torch.cat((self.Y, CUDA(torch.tensor(label).float())), dim=0)

    # NOTICE: process data, not used here
    # def split_train_validation(self):
    #     num_data = len(self.data)
    #     # NOTICE: Not sure if we should normalize data for NP
    #     # normalization, note that we should not overwrite the original data and label
    #     # self.mu = torch.mean(self.data, dim=0, keepdims=True)
    #     # self.sigma = torch.std(self.data, dim=0, keepdims=True)
    #     # self.train_data = (self.data - self.mu) / self.sigma
    #     #
    #     # self.label_mu = torch.mean(self.label, dim=0, keepdims=True)
    #     # self.label_sigma = torch.std(self.label, dim=0, keepdims=True)
    #     # self.train_label = (self.label - self.label_mu) / self.label_sigma
    #     #
    #     self.train_data = self.data
    #     self.train_label = self.label
    #
    #     # use validation, no validation for NP model now
    #     if self.validation_flag:
    #         indices = list(range(num_data))
    #         split = int(np.floor(self.validation_ratio * num_data))
    #         np.random.shuffle(indices)
    #         train_idx, test_idx = indices[split:], indices[:split]
    #
    #         train_set = [[self.train_data[idx], self.train_label[idx]] for idx in train_idx]
    #         test_set = [[self.train_data[idx], self.train_label[idx]] for idx in test_idx]
    #
    #         train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=self.batch_size)
    #         test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=self.batch_size)
    #     else:
    #         train_set = [[self.train_data[idx], self.train_label[idx]] for idx in range(num_data)]
    #         train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=self.batch_size)
    #         test_loader = None
    #
    #     return train_loader, test_loader

    # NOTICE:
    #   prepare data for NP,
    #   return dataloader
    def get_train_data(self):
        '''
        return a daraloader
        :return: for ctt_x, ctt_y, tgt_x, tgt_y in dataloader
                     ctt_x (batch, context_number + target+number, x_dim)
                     ctt_y (batch, context_number + target+number, y_dim)
                     ctt_x (batch, target+number, x_dim)
                     ctt_y (batch, target+number, y_dim)
        '''

        # Notice: May need to normalize
        # TODO: normalize
        time_start = time.time()

        data_num = self.X.size()[0]
        # Notice: permute the training data for context
        if not self.sequential:
            ind = torch.randperm(data_num)
            X_train_tgt = self.X[ind, :]
            Y_train_tgt = self.Y[ind, :]
        else:
            X_train_tgt = self.X
            Y_train_tgt = self.Y

        # Notice: permute the training data for target
        if not self.sequential:
            ind_ctt = torch.randperm(data_num)
            X_train_ctt = self.X[ind_ctt, :]
            Y_train_ctt = self.Y[ind_ctt, :]
        else:
            X_train_ctt = self.X
            Y_train_ctt = self.Y

        # TODO: Deal the situation when *data_num <= self.context_number*
        # FIXME: randint error

        # Notice: get the start index for seq for context
        if data_num > self.context_number:
            ctt_start_index = torch.randperm(data_num - self.context_number)
            ctt_index_select = torch.randint(0, data_num - self.context_number, (data_num - self.target_number + 1, ))

            loader_test = []
            for i in range(data_num - self.target_number + 1):
                # Notice: get target (sequences) *principle, all data are encouraged to serve as target
                tgt_x = X_train_tgt[i:i + self.target_number, :]
                tgt_y = Y_train_tgt[i:i + self.target_number, :]

                # NOTICE: get context (sequences) context will include target
                start_index = ctt_start_index[ctt_index_select[i]]
                ctt_x = X_train_ctt[start_index: start_index + self.context_number, :]
                ctt_x = torch.cat((tgt_x, ctt_x), dim=0)
                ctt_y = Y_train_ctt[start_index: start_index + self.context_number, :]
                ctt_y = torch.cat((tgt_y, ctt_y), dim=0)

                # NOTICE: get data
                temp = [ctt_x, ctt_y, tgt_x, tgt_y]
                loader_test.append(temp)

            loader = torch.utils.data.DataLoader(loader_test, batch_size=self.batch_size)

        elif data_num <= self.context_number:
            ctt_index = torch.randperm(data_num)
            loader_test = []
            for i in range(data_num):
                # Notice: get target (sequences)
                #   There are two cases: 1) data num <= self.target
                if data_num <= self.target_number:
                    tgt_x = X_train_tgt[:, :]
                    tgt_y = Y_train_tgt[:, :]
                elif data_num > self.target_number:
                    X_train_tgt_duplicated = X_train_tgt.repeat((10, 1))
                    Y_train_tgt_duplicated = Y_train_tgt.repeat((10, 1))
                    tgt_x = X_train_tgt_duplicated[i:i + self.target_number, :]
                    tgt_y = Y_train_tgt_duplicated[i:i + self.target_number, :]

                # Notice: get context (sequences) context will include target
                #   Since data_num is smaller than context number, context will always cover target
                ctt_x = X_train_ctt
                ctt_y = Y_train_ctt

                # NOTICE: get data
                temp = [ctt_x, ctt_y, tgt_x, tgt_y]
                loader_test.append(temp)

            loader = torch.utils.data.DataLoader(loader_test, batch_size=self.batch_size)

        end_time = time.time()
        print('Time used in get train data for NP', end_time - time_start)
        return loader
    
    # NOTICE: Fit
    def fit(self, data=None):
        # NOTICE: tunable parameter or methods
        #   self.epoch
        #   self.sequential = True or False
        #   self.virtual_batch = False

        self.model.train()

        print('fitting model')
        # NOTICE: Not know how to .fit() with input data
        if data is not None:
            self.data_process(data)
        print('Data number is ', self.X.size()[0])
        # NOTICE:
        #   Use different strategy 'nn' and 'naive'
        #   'nn', during each epoch, only use a part of data
        #   Using Naive is extremely slow, because it doesn't have batchsize, and the number of context
        #   is increasing
        if self.strategy == 'naive':
            X = self.X      # (data_num, x_dim)
            Y = self.Y      # (data_num, y_dim)

            context_x = X.view((1, -1, self.x_dim))  # (1, data_num, x_dim)
            context_y = Y.view((1, -1, self.y_dim))  # (1, data_num, y_dim)
            target_x = context_x  # (1, data_num, x_dim)
            target_y = context_y  # (1, data_num, y_dim)
            data_num = len(X)  # obtain the num of data (sequence length)

            for epoch in range(self.n_epochs):
                # NOTICE: do not consider the sequential order in data, then randomly permutate them
                if not self.sequential:
                    # NOTICE: permutate X and Y respectively in each step
                    rand_ind_ctt = torch.randperm(data_num)
                    rand_ind_tgt = torch.randperm(data_num)
                    context_x = context_x[:, rand_ind_ctt, :]
                    context_y = context_y[:, rand_ind_ctt, :]
                    target_x = target_x[:, rand_ind_tgt, :]
                    target_y = target_y[:, rand_ind_tgt, :]

                # NOTICE: forward
                self.optim.zero_grad()
                mu, sigma, log_p, kl, loss = self.model(context_x, context_y, target_x, target_y)
                loss.backward()
                self.optim.step()
                if logger:
                    logger.info(f"training epoch [{epoch}/{self.n_epochs}],loss train: {loss:.4f}.")

        elif self.strategy == 'nn':
            # NOTICE: process the data to get iterable ctt and tgt
            train_loader = self.get_train_data()

            for epoch in range(self.n_epochs):
                loss_epoch = []
                for context_x, context_y, target_x, target_y in train_loader:
                    self.optim.zero_grad()
                    mu, sigma, log_p, kl, loss = self.model(context_x, context_y, target_x, target_y)
                    loss.backward()

                    loss_epoch.append(loss.item())
                    self.optim.step()

                if logger:
                    logger.info(f"training epoch [{epoch}/{self.n_epochs}],"
                                f"loss train: {np.mean(loss_epoch):.4f}.")

        elif self.strategy == 'deepmind':
            # Notice: In each iteration(epoch), randomly sample a number of contexts
            #   and targets, meanwhile, the context should always include target


            for epoch in range(self.n_epochs):
                loss_epoch = []
                # Notice: get context


            return

        return loss.item()

    # NOTICE: Completely rewrite it
    def likelihood(self, dataset):
        self.model.eval()
        # dataset : list of [task_idx, state, action, next_state-state]
        # input_data_num = len(dataset)  # obtain the num of data (sequence length)

        # convert list of data into X and Y tensor
        input_x_tensor, input_y_tensor = self.list_2_torch(dataset)  # (input_data_num, x_dim), (input_data_num, y_dim)
        target_x = CUDA(input_x_tensor.view((1, -1, self.x_dim)))  # (1, input_data_num, x_dim)
        target_y = CUDA(input_y_tensor.view((1, -1, self.y_dim)))  # (1, input_data_num, y_dim)

        if self.likelihood_method == 'gp':
            # NOTICE: Use all the existing data as context
            # exist_data_num = len(self.dataset)
            exist_x_tensor, exist_y_tensor = self.list_2_torch(self.dataset)
            context_x = CUDA(exist_x_tensor.view(1, -1, self.x_dim))
            context_y = CUDA(exist_y_tensor.view(1, -1, self.y_dim))
        elif self.likelihood_method == 'nn':
            context_x = target_x
            context_y = target_y
        else:
            print('Please choose likelihood method')

        # NOTICE: forward the NP
        mu, sigma, log_p, kl, loss = self.model(context_x, context_y, target_x, target_y)

        if self.likelihood_value == 'loss':
            # NOTICE: Use negative loss to represent the likelihood F(input_data, model)
            return - loss.cpu().detach().numpy()

        elif self.likelihood_value == 'll':
            # NOTICE: Use the sum of the likelihood of N(data | pred_mu, pred_var)
            mu_dist_v = mu.view(-1, self.y_dim)  # (test_num, y_dim)
            sigma_dist_v = sigma.view(-1, self.y_dim)  # (test_num, y_dim)
            cov = torch.diag_embed(sigma_dist_v)  # (test_num, y_dim, y_dim)
            mg = torch.distributions.MultivariateNormal(mu_dist_v, cov)
            ll = mg.log_prob(target_y.view((-1, self.y_dim)))
            ll_sum = torch.sum(ll)
            return ll_sum.cpu().detach().numpy()

    # NOTICE: Predict
    # TODO: forward the NP to make the prediction
    def predict(self, s, a):
        # convert to torch format
        self.model.eval()
        s = CUDA(torch.tensor(s).float())
        a = CUDA(torch.tensor(a).float())
        inputs = torch.cat((s, a), axis=1)  # (1, x_dim)
        #print('inputs.size() =', inputs.size())
        target_x = inputs.view((1, -1, self.x_dim))

        # NOTICE: Forward NP to generate target_y
        if self.strategy == 'nn':
            # Notice: permute or not
            data_num = self.X.size()[0]
            # print('Current data_num is', data_num)
            if not self.sequential:
                ind_ctt = CUDA(torch.randperm(data_num))
                all_existing_X = self.X[ind_ctt, :]
                all_existing_Y = self.Y[ind_ctt, :]
            else:
                all_existing_X = self.X
                all_existing_Y = self.Y

            # Notice: there are 2 cases:
            #   1) data_num > self.context_number
            #   2) data_num <= self.context_number
            if data_num > self.context_number:
                # sample a random portion of data as context
                ctt_index = np.random.randint(0, data_num - self.context_number)
                context_x_tensor = CUDA(all_existing_X[ctt_index:ctt_index + self.context_number, :])
                context_y_tensor = CUDA(all_existing_Y[ctt_index:ctt_index + self.context_number, :])

            elif data_num <= self.context_number:
                # ctt_index = 0
                # lefted_context_number = data_num
                context_x_tensor = CUDA(all_existing_X[:, :])
                context_y_tensor = CUDA(all_existing_Y[:, :])
            else:
                print("Wrong in predict, 'nn' strategy. ")

            context_x = context_x_tensor.view(1, -1, self.x_dim)
            context_y = context_y_tensor.view(1, -1, self.y_dim)

            # NOTICE: forward the NP
            mu, sigma, log_p, kl, loss = self.model(context_x, context_y, target_x, None)

        # TODO: not sure about the output dim of mu and sigma
        mu = torch.squeeze(mu, 0)
        sigma = torch.squeeze(sigma, 0)
        return mu.cpu().detach().numpy()

    # Notice: test function used for investigating MSE loss of prediction
    def test(self, s, a, x_g):
        # convert to torch format
        self.model.eval()

        # Notice, directly use predict(self)
        mu = self.predict(s, a)

        # s = CUDA(torch.tensor(s).float())
        # a = CUDA(torch.tensor(a).float())
        # inputs = torch.cat((s, a), axis=1)  # (1, x_dim)
        # #print('inputs.size() =', inputs.size())
        # target_x = inputs.view((1, -1, self.x_dim))
        #
        # # NOTICE: Forward NP to generate target_y
        # #  use the existing data as context
        # exist_x_tensor = self.X[-500:, :]
        # exist_y_tensor = self.Y[-500:, :]
        # # exist_x_tensor, exist_y_tensor = self.list_2_torch(self.dataset)
        # # NOTICE: if use all the existing data, the memory will explode..
        # #   try select a number of data say 500 ?
        #
        # context_x = CUDA(exist_x_tensor.view(1, -1, self.x_dim))
        # context_y = CUDA(exist_y_tensor.view(1, -1, self.y_dim))
        #
        # # NOTICE: forward the NP
        # mu, sigma, log_p, kl, loss = self.model(context_x, context_y, target_x, None)
        #
        # # TODO: not sure about the output dim of mu and sigma
        # mu = torch.squeeze(mu, 0)
        # sigma = torch.squeeze(sigma, 0)
        # mu = mu.cpu().detach().numpy()

        mse_error = np.sum((mu-x_g)**2)
        return mu, mse_error


if __name__ == '__main__':
    model = NP()
