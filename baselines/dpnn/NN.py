
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from loguru import logger
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


def CUDA(var):
    return var.cuda() if torch.cuda.is_available() else var

class MLP(nn.Module):
    def __init__(self, n_input=7, n_output=6, n_h=2, size_h=128):
        super(MLP, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.fc_in = nn.Linear(n_input, size_h)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc_list = nn.ModuleList()
        for i in range(n_h - 1):
            self.fc_list.append(nn.Linear(size_h, size_h))
        
        #self.fc_out = nn.Linear(size_h, n_output)
        self.fc_out_mean = nn.Linear(size_h, n_output)
        self.fc_out_var = nn.Linear(size_h, n_output)
        
        # Initialize weight
        nn.init.normal_(self.fc_in.weight, 0.0, 0.02)
        nn.init.normal_(self.fc_out_mean.weight, 0.0, 0.02)
        nn.init.normal_(self.fc_out_var.weight, 0.0, 0.02)
        
        self.fc_list.apply(self.init_normal)
        
    def init_normal(self, m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x):
        out = x.view(-1, self.n_input)
        out = self.fc_in(out)
        out = self.relu(out)
        for _, layer in enumerate(self.fc_list, start=0):
            out = layer(out)
            out = self.relu(out)
        out_mean = self.fc_out_mean(out)
        out_var = self.fc_out_var(out)
        out_var = self.relu(out_var)
        out_var = out_var + 0.001 # add a small bias to make sure it is not equal to 0
        return (out_mean, out_var)


class NNComponent(object):
    # output: [state mean, state var]
    name = "NN"
    
    def __init__(self, NN_config):
        super().__init__()
        model_config = NN_config["model_config"]
        training_config = NN_config["training_config"]
        
        self.state_dim = model_config["state_dim"]
        self.action_dim = model_config["action_dim"]
        self.input_dim = self.state_dim+self.action_dim

        self.n_epochs = training_config["n_epochs"]
        self.lr = training_config["learning_rate"]
        self.batch_size = training_config["batch_size"]
        self.save_model_path = model_config["save_model_path"]
        
        self.validation_flag = training_config["validation_flag"]
        self.validate_freq = training_config["validation_freq"]
        self.validation_ratio = training_config["validation_ratio"]

        self.model = CUDA(MLP(self.input_dim, self.state_dim, model_config["hidden_dim"], model_config["hidden_size"]))
        self.mse = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        #self.data = None
        #self.label = None
        self.dataset = []

    def criterion(self, output, label):
        nll = -self.log_likelihood(output, label) # [batch]
        return torch.mean(nll)

    def log_likelihood(self, output, label):
        mu = output[0] # [batch, state_dim]
        var = output[1] # [batch, state_dim]
        cov = torch.diag_embed(var) # [batch, state_dim, state_dim]
        m = MultivariateNormal(mu, cov)
        ll = m.log_prob(label) # [batch]
        return ll

    def likelihood(self, dataset):
        # dataset format: list of [task_idx, state, action, next_state-state]

        data_list = self.process_dataset(dataset)
        data_loader = torch.utils.data.DataLoader(data_list, shuffle=False, batch_size=len(data_list))
        l = 0
        for datas, labels in data_loader:
            outputs = self.model(datas)
            ll = self.log_likelihood(outputs, labels) # [batch]
            #print("ll: ", ll)
            #print("exp ll: ", torch.exp(ll))
            l = torch.mean( torch.exp(ll) ).item()
            #print("l: ", l)
        return l

    def process_dataset(self, dataset):
        # dataset format: list of [task_idx, state, action, next_state-state]
        data_list = []
        for data in dataset:
            s = data[1] # state
            a = data[2] # action
            label = data[3] # here label means the next state [state dim]
            data = np.concatenate((s, a), axis=0) # [state dim + action dim]
            data_torch = CUDA(torch.Tensor(data))
            label_torch = CUDA(torch.Tensor(label))
            data_list.append([data_torch, label_torch])
        return data_list

    def predict(self, s, a):
        # convert to torch format
        s = CUDA(torch.tensor(s).float())
        a = CUDA(torch.tensor(a).float())
        inputs = torch.cat((s, a), axis=1)
        mean, var = self.model(inputs)
        state_next = mean.cpu().detach().numpy()
        return state_next, var.cpu().detach().numpy()
    
    def add_data_point(self, data):
        # data format: [task_idx, state, action, next_state-state]
        self.dataset.append(data)
        
    def reset_dataset(self, new_dataset = None):
        # dataset format: list of [task_idx, state, action, next_state-state]
        if new_dataset is not None:
            self.dataset = new_dataset
        else:
            self.dataset = []
            
    def make_dataset(self, dataset, make_test_set=False):
        # dataset format: list of [task_idx, state, action, next_state-state]
        num_data = len(dataset)
        data_list = self.process_dataset(dataset)
            
        if make_test_set:
            indices = list(range(num_data))
            split = int(np.floor(self.validation_ratio * num_data))
            np.random.shuffle(indices)
            train_idx, test_idx = indices[split:], indices[:split]
            train_set = [data_list[idx] for idx in train_idx]
            test_set = [data_list[idx] for idx in test_idx]
            train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=self.batch_size)
            test_loader = None
            if len(test_set):
                test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=self.batch_size)
        else:
            train_loader = torch.utils.data.DataLoader(data_list, shuffle=True, batch_size=self.batch_size)
            test_loader = None
        return train_loader, test_loader

    def fit(self, dataset=None, logger=True):
        if dataset is not None:
            train_loader, test_loader = self.make_dataset(dataset, make_test_set=self.validation_flag)
        else: # use its own accumulated data
            train_loader, test_loader = self.make_dataset(self.dataset, make_test_set=self.validation_flag)
        
        for epoch in range(self.n_epochs):
            loss_this_epoch = []
            for datas, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(datas)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                loss_this_epoch.append(loss.item())

            if self.validation_flag and (epoch+1) % self.validate_freq == 0:
                loss_test, mse_test = np.inf, np.inf
                if test_loader is not None:
                    loss_test, mse_test = self.validate_model(test_loader)
                loss_train, mse_train = self.validate_model(train_loader)
                if logger:
                    print(f"training epoch [{epoch}/{self.n_epochs}],loss train: {loss_train:.4f}/{mse_train:.4f}, loss test  {loss_test:.4f}/{mse_test:.4f}")

        return np.mean(loss_this_epoch)

    def validate_model(self, testloader):
        loss_list = []
        mse_list = []
        for datas, labels in testloader:
            outputs = self.model(datas)
            loss = self.criterion(outputs, labels)
            mse_loss = self.mse(outputs[0], labels)
            loss_list.append(loss.item())
            mse_list.append(mse_loss.item())
        return np.mean(loss_list), np.mean(mse_list)

    def reset_model(self):
        def weight_reset(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, 0.0, 0.02)
        self.model.apply(weight_reset)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
    
    def split_train_validation_old(self):
        num_data = len(self.data)
        # use validation
        if self.validation_flag:
            indices = list(range(num_data))
            split = int(np.floor(self.validation_ratio * num_data))
            np.random.shuffle(indices)
            train_idx, test_idx = indices[split:], indices[:split]

            train_set = [[self.data[idx], self.label[idx]] for idx in train_idx]
            test_set = [[self.data[idx], self.label[idx]] for idx in test_idx]

            train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=self.batch_size)
            test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=self.batch_size)
        else:
            train_set = [[self.data[idx], self.label[idx]] for idx in range(num_data)]
            train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=self.batch_size)
            test_loader = None
            
        return train_loader, test_loader

    def add_data_point_old(self, data):
        s = data[1]
        a = data[2]
        label = data[3][None] # here label means the next state
        data = np.concatenate((s, a), axis=0)[None]

        # add new data point to data buffer
        if self.data is None:
            self.data = CUDA(torch.Tensor(data))
            self.label = CUDA(torch.Tensor(label))
        else:
            self.data = torch.cat((self.data, CUDA(torch.tensor(data).float())), dim=0)
            self.label = torch.cat((self.label, CUDA(torch.tensor(label).float())), dim=0)
