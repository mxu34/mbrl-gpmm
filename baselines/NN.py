

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
        nn.init.normal_(self.fc_in.weight, 0.0, 0.02)
        nn.init.normal_(self.fc_out.weight, 0.0, 0.02)
        self.fc_list.apply(self.init_normal)

    def init_normal(self, m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x):
        out = x.view(-1, self.n_input)
        out = self.fc_in(out)
        out = self.tanh(out)
        for _, layer in enumerate(self.fc_list, start=0):
            out = layer(out)
            out = self.tanh(out)
        out = self.fc_out(out)

        #out = self.tanh(out)
        return out


class NN(object):
    name = 'NN'
    
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
        
        self.save_model_flag = model_config["save_model_flag"]
        self.save_model_path = model_config["save_model_path"]
        
        self.validation_flag = training_config["validation_flag"]
        self.validate_freq = training_config["validation_freq"]
        self.validation_ratio = training_config["validation_ratio"]

        if model_config["load_model"]:
            self.model = CUDA(torch.load(model_config["save_model_path"]))
        else:
            self.model = CUDA(MLP(self.input_dim, self.state_dim, model_config["hidden_dim"], model_config["hidden_size"]))

        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.data = None
        self.label = None
        self.mu = CUDA(torch.tensor(0.0))
        self.sigma = CUDA(torch.tensor(1.0))
        self.label_mu = CUDA(torch.tensor(0.0))
        self.label_sigma = CUDA(torch.tensor(1.0))
        self.eps = 1e-7

    def reset_model(self):
        def weight_reset(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, 0.0, 0.02)
        self.model.apply(weight_reset)

    def data_process(self, data):
        s = data[1][None]
        a = data[2][None]
        label = data[3][None] # here label means the next state
        data = np.concatenate((s, a), axis=1)

        # add new data point to data buffer
        if self.data is None:
            self.data = CUDA(torch.Tensor(data))
            self.label = CUDA(torch.Tensor(label))
        else:
            self.data = torch.cat((self.data, CUDA(torch.tensor(data).float())), dim=0)
            self.label = torch.cat((self.label, CUDA(torch.tensor(label).float())), dim=0)

    def split_train_validation(self):
        num_data = len(self.data)

        # normalization, note that we should not overrite the original data and label
        self.mu = torch.mean(self.data, dim=0, keepdims=True)
        self.sigma = torch.std(self.data, dim=0, keepdims=True)
        self.train_data = (self.data-self.mu) / self.sigma

        self.label_mu = torch.mean(self.label, dim=0, keepdims=True)
        self.label_sigma = torch.std(self.label, dim=0, keepdims=True)
        self.train_label = (self.label-self.label_mu) / self.label_sigma

        # use validation
        if self.validation_flag:
            indices = list(range(num_data))
            split = int(np.floor(self.validation_ratio * num_data))
            np.random.shuffle(indices)
            train_idx, test_idx = indices[split:], indices[:split]

            train_set = [[self.train_data[idx], self.train_label[idx]] for idx in train_idx]
            test_set = [[self.train_data[idx], self.train_label[idx]] for idx in test_idx]

            train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=self.batch_size)
            test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=self.batch_size)
        else:
            train_set = [[self.train_data[idx], self.train_label[idx]] for idx in range(num_data)]
            train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=self.batch_size)
            test_loader = None
            
        return train_loader, test_loader

    def fit(self, data=None):
        print('data size: ', len(self.data))
        if data is not None:
            self.data_process(data)
        train_loader, test_loader = self.split_train_validation()
        #self.reset_model()
        best_test_loss = np.inf
        
        for epoch in range(self.n_epochs):
            loss_this_epoch = []
            for datas, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(datas)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                loss_this_epoch.append(loss.item())
            
            if self.save_model_flag:
                torch.save(self.model.state_dict(), self.save_model_path)
                
            if self.validation_flag and (epoch+1) % self.validate_freq == 0:
                loss_test = self.validate_model(test_loader)
                if best_test_loss > loss_test:
                    best_test_loss = loss_test
                    best_model = self.model.state_dict()
                logger.info(f"Epoch [{epoch}/{self.n_epochs}], loss train: {np.mean(loss_this_epoch):.4f}, loss test  {loss_test:.4f}")

        self.model.load_state_dict(best_model)
        return np.mean(loss_this_epoch)

    def validate_model(self, testloader):
        loss_list = []
        for datas, labels in testloader:
            outputs = self.model(datas)
            loss = self.criterion(outputs, labels)
            loss_list.append(loss.item())
        return np.mean(loss_list)

    def predict(self, s, a):
        # convert to torch format
        s = CUDA(torch.tensor(s).float())
        a = CUDA(torch.tensor(a).float())
        inputs = torch.cat((s, a), axis=1)
        inputs = (inputs-self.mu) / (self.sigma + self.eps)
        with torch.no_grad():
            ds = self.model(inputs)
            ds = ds * (self.label_sigma + self.eps) + self.label_mu
            ds = ds.cpu().detach().numpy()
        return ds

    def test(self, s, a, x_g):
        # convert to torch format
        s = CUDA(torch.tensor(s).float())
        a = CUDA(torch.tensor(a).float())
        inputs = torch.cat((s, a), axis=1)
        inputs = (inputs-self.mu) / (self.sigma + self.eps)
        with torch.no_grad():
            ds = self.model(inputs)
            ds_unnormal = ds * (self.label_sigma + self.eps) + self.label_mu
            ds_unnormal = ds_unnormal.cpu().detach().numpy()
            ds = ds.cpu().detach().numpy()
        
        x_g = CUDA(torch.Tensor(x_g))
        x_g = (x_g-self.label_mu) / (self.label_sigma + self.eps)
        x_g = x_g.cpu().numpy()
        mse_error = np.sum((ds-x_g)**2)

        return ds_unnormal, mse_error
