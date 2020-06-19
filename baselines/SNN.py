

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn

from loguru import logger


def CUDA(var):
    return var.cuda() if torch.cuda.is_available() else var


class MLP(nn.Module):
    def __init__(self, n_input=7, n_output=6, n_h=2, size_h=128):
        super(MLP, self).__init__()
        self.n_input = n_input
        self.fc_in = nn.Linear(n_input, size_h)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc_list = nn.ModuleList()
        for i in range(n_h - 1):
            self.fc_list.append(nn.Linear(size_h, size_h))
        self.fc_out = nn.Linear(size_h, n_output)
        
        # Initialize weight
        nn.init.uniform_(self.fc_in.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_out.weight, -0.1, 0.1)
        self.fc_list.apply(self.init_normal)

    def init_normal(self, m):
        if type(m) == nn.Linear:
            nn.init.uniform_(m.weight, -0.1, 0.1)

    def forward(self, x):
        out = x.view(-1, self.n_input)
        out = self.fc_in(out)
        out = self.tanh(out)
        for _, layer in enumerate(self.fc_list, start=0):
            out = layer(out)
            out = self.tanh(out)
        out = self.fc_out(out)
        return out


class SNN(object):
    name = 'SNN'
    # use two seprate NN to model gripper and object
    
    def __init__(self, NN_config):
        super().__init__()
        model_config = NN_config["model_config"]
        training_config = NN_config["training_config"]
        
        self.state_dim = int(model_config["state_dim"]/2)
        self.action_dim = model_config["action_dim"]
        self.input_dim = self.state_dim+self.action_dim

        self.n_epochs = training_config["n_epochs"]
        self.lr = training_config["learning_rate"]
        self.batch_size = training_config["batch_size"]
        
        self.save_model_flag = training_config["save_model_flag"]
        self.save_model_path = training_config["save_model_path"]
        
        self.validation_flag = training_config["validation_flag"]
        self.validate_freq = training_config["validation_freq"]
        self.validation_ratio = training_config["validation_ratio"]

        self.model_shared = CUDA(MLP(self.state_dim+self.action_dim, self.state_dim, model_config["hidden_dim"], model_config["hidden_size"]))
        self.model_indep = CUDA(MLP(self.state_dim, self.state_dim, model_config["hidden_dim"], model_config["hidden_size"]))

        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer_shared = torch.optim.Adam(self.model_shared.parameters(), lr=self.lr)
        self.optimizer_indep = torch.optim.Adam(self.model_indep.parameters(), lr=self.lr)

        self.data = None
        self.label = None

    def predict(self, s, a):
        # convert to torch format
        s = CUDA(torch.tensor(s).float())
        a = CUDA(torch.tensor(a).float())

        inputs = torch.cat((s[:, :self.state_dim], a, s[:, self.state_dim:]), axis=1)
        state_next_1 = self.model_shared(inputs[:, :self.state_dim+self.action_dim]).cpu().detach().numpy()
        state_next_2 = self.model_indep(inputs[:, self.state_dim+self.action_dim:]).cpu().detach().numpy()

        state_next = np.concatenate((state_next_1, state_next_2), axis=1)
        return state_next

    def reset_model(self):
        def weight_reset(m):
            if type(m) == nn.Linear:
                nn.init.uniform_(m.weight, -1, 1)
        self.model.apply(weight_reset)

    def data_process(self, data):
        s = data[1]
        a = data[2]
        label = data[3][None] # here label means the next state
        data = np.concatenate((s[:self.state_dim], a, s[self.state_dim:]), axis=0)[None]

        # add new data point to data buffer
        if self.data is None:
            self.data = CUDA(torch.Tensor(data))
            self.label = CUDA(torch.Tensor(label))
        else:
            self.data = torch.cat((self.data, CUDA(torch.tensor(data).float())), dim=0)
            self.label = torch.cat((self.label, CUDA(torch.tensor(label).float())), dim=0)

    def split_train_validation(self):
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

    def fit(self, data=None):
        if data is not None:
            self.data_process(data)
        train_loader, test_loader = self.split_train_validation()
        
        for epoch in range(self.n_epochs):
            loss_this_epoch = []
            for datas, labels in train_loader:
                # train shared model
                self.optimizer_shared.zero_grad()
                outputs = self.model_shared(datas[:, 0:self.state_dim+self.action_dim])
                loss = self.criterion(outputs, labels[:, 0:self.state_dim])
                loss.backward()
                self.optimizer_shared.step()

                # train independent model
                self.optimizer_indep.zero_grad()
                outputs = self.model_indep(datas[:, self.state_dim+self.action_dim:])
                loss = self.criterion(outputs, labels[:, self.state_dim:])
                loss.backward()
                self.optimizer_indep.step()

                loss_this_epoch.append(loss.item())
            
            if self.save_model_flag:
                torch.save(self.model, self.save_model_path)
                
            if self.validation_flag and (epoch+1) % self.validate_freq == 0:
                loss_test = self.validate_model(test_loader)
                logger.info(f"training epoch [{epoch}/{self.n_epochs}], loss train: {np.mean(loss_this_epoch):.4f}, loss test  {loss_test:.4f}")

        return np.mean(loss_this_epoch)

    def validate_model(self, testloader):
        loss_list = []
        for datas, labels in testloader:
            outputs = self.model(datas)
            loss = self.criterion(outputs, labels)
            loss_list.append(loss.item())
        return np.mean(loss_list)
