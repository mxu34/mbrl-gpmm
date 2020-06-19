

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
import random


def CUDA(var):
    return var.cuda() if torch.cuda.is_available() else var


class ModuleWrapper(nn.Module):
    def params(self):
        return [p for _, p in self.named_params()]
    
    def named_leaves(self):
        return []
    
    def named_submodules(self):
        return []
    
    def named_params(self):
        subparams = []
        for name, mod in self.named_submodules():
            for subname, param in mod.named_params():
                subparams.append((name + '.' + subname, param))
        return self.named_leaves() + subparams
    
    def set_param(self, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in self.named_submodules():
                if module_name == name:
                    mod.set_param(rest, param)
                    break
        else:
            setattr(self, name, param)
            
    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = V(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class GradLinear(ModuleWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = CUDA(nn.Linear(*args, **kwargs))
        self.weights = V(ignore.weight.data, requires_grad=True)
        self.bias = V(ignore.bias.data, requires_grad=True)

    def forward(self, x):
        return F.linear(x, self.weights, self.bias)

    def named_leaves(self):
        return [('weights', self.weights), ('bias', self.bias)]


class MetaModel(ModuleWrapper):
    def __init__(self, state_dim, action_dim, hidden_size):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = self.state_dim + self.action_dim
        self.hidden_size = hidden_size

        self.hidden1 = GradLinear(self.input_dim, self.hidden_size)
        self.hidden2 = GradLinear(self.hidden_size, self.hidden_size)
        self.out = GradLinear(self.hidden_size, self.state_dim)
        self.init_parameters()

    def forward(self, x):
        x = F.tanh(self.hidden1(x))
        x = F.tanh(self.hidden2(x))
        return self.out(x)
    
    def init_parameters(self):
        nn.init.normal_(self.hidden1.weights, 0.0, 0.02)
        nn.init.normal_(self.hidden2.weights, 0.0, 0.02)
        nn.init.normal_(self.out.weights, 0.0, 0.02)
        nn.init.constant_(self.hidden1.bias, 0.0)
        nn.init.constant_(self.hidden2.bias, 0.0)
        nn.init.constant_(self.out.bias, 0.0)
        
    def named_submodules(self):
        return [('hidden1', self.hidden1), ('hidden2', self.hidden2), ('out', self.out)]


class MAML(object):
    name = 'MAML'

    def __init__(self, maml_config):
        super().__init__()
        self.state_dim = maml_config['state_dim']
        self.action_dim = maml_config['action_dim']
        self.adapt_iter = maml_config['adapt_iter']
        self.alpha = maml_config['alpha']
        self.adapt_lr = maml_config['adapt_lr']
        self.meta_lr = maml_config['meta_lr']
        self.meta_epoch = maml_config['meta_epoch']
        self.meta_batch_size = maml_config['meta_batch_size']
        self.hidden_size = maml_config['hidden_size']

        self.meta_model = CUDA(MetaModel(self.state_dim, self.action_dim, self.hidden_size))
        self.adapt_model = CUDA(MetaModel(self.state_dim, self.action_dim, self.hidden_size))
        self.meta_optimizer = torch.optim.Adam(self.meta_model.params(), lr=self.meta_lr)

        # we need to store data according to their task boundary
        self.meta_data_collector = []
        self.meta_label_collector = []

        self.meta_data = None
        self.meta_label = None
        self.adapt_data = None
        self.adapt_label = None

    def data_process(self, data, used_for_adaption=False):
        s = data[1][None]
        a = data[2][None]
        label = data[3][None] # here label means the next state
        data = np.concatenate((s, a), axis=1)

        # add new data point to data buffer
        if not used_for_adaption:
            if self.meta_data is None:
                self.meta_data = CUDA(torch.Tensor(data))
                self.meta_label = CUDA(torch.Tensor(label))
            else:
                self.meta_data = torch.cat((self.meta_data, CUDA(torch.tensor(data).float())), dim=0)
                self.meta_label = torch.cat((self.meta_label, CUDA(torch.tensor(label).float())), dim=0)

        # these data should also be used for adaption
        if used_for_adaption:
            if self.adapt_data is None:
                self.adapt_data = CUDA(torch.Tensor(data))
                self.adapt_label = CUDA(torch.Tensor(label))
            else:
                self.adapt_data = torch.cat((self.adapt_data, CUDA(torch.tensor(data).float())), dim=0)
                self.adapt_label = torch.cat((self.adapt_label, CUDA(torch.tensor(label).float())), dim=0)

    @staticmethod
    def inner_fit(model, x, y, create_graph=False):
        model.train()
        loss = F.mse_loss(model(x), y)
        loss.backward(create_graph=create_graph, retain_graph=True)
        return loss.data.cpu().numpy()

    @staticmethod
    def normalize(x, mean, std):
        return (x-mean)/std
        
    @staticmethod
    def denormalize(x, mean, std):
        return x*std+mean

    def calculate_normalization(self):
        episode_num = len(self.meta_data_collector)
        data = torch.cat([self.meta_data_collector[i] for i in range(episode_num)])
        label = torch.cat([self.meta_label_collector[i] for i in range(episode_num)])

        # normalization, note that we should not overrite the original data and label
        self.data_mu = torch.mean(data, dim=0, keepdims=True)
        self.data_sigma = torch.std(data, dim=0, keepdims=True)
        self.label_mu = torch.mean(label, dim=0, keepdims=True)
        self.label_sigma = torch.std(label, dim=0, keepdims=True)

        data_collector = [self.normalize(self.meta_data_collector[i], self.data_mu, self.data_sigma) for i in range(episode_num)]
        label_collector = [self.normalize(self.meta_label_collector[i], self.label_mu, self.label_sigma) for i in range(episode_num)]
        
        return data_collector, label_collector

    # In this function, we equally divide one trajectory into two parts (after randomly shuffle the trajectories)
    # one is for updating adaptive model, the other is for updating meta model
    # However, the first part and second part data may not be consistent
    def fit(self):
        data_collector, label_collector = self.calculate_normalization()
        #data_collector, label_collector = self.meta_data_collector, self.meta_label_collector
        # reset meta model
        self.meta_model.init_parameters()
        #self.adapt_model.init_parameters()
        episode_num = len(data_collector)
        print('Training MAML with number of episode: ', episode_num)
        for _ in range(self.meta_epoch):
            for t_i, r_i in enumerate(random.sample(range(episode_num), episode_num)):
                # split the data, half for train meta, half for adapt
                x = data_collector[r_i]
                y = label_collector[r_i]

                # shuffle the original data
                rand_index = torch.randperm(x.shape[0])
                x = x[rand_index]
                y = y[rand_index]

                # split to two datasets
                split_len = (int(x.shape[0]/2), x.shape[0]-int(x.shape[0]/2))
                x_train, x_test = torch.split(x, split_len)
                y_train, y_test = torch.split(y, split_len)
                
                # [Note] Here we use the same parameter of the model even we call copy !!
                self.adapt_model.copy(self.meta_model, same_var=True)
                self.inner_fit(self.adapt_model, x_train, y_train, create_graph=True)
                for name, param in self.adapt_model.named_params():
                    grad = param.grad
                    self.adapt_model.set_param(name, param-self.alpha*grad) # lr_inner -> alpha in paper
                
                # use new training data to accumulate the gradient since we only call .backward()
                loss = self.inner_fit(self.adapt_model, x_test, y_test)

                # update the parameter of meta-model
                # if we dont call this optimizer, the gradient will be accumulated
                if (t_i + 1) % self.meta_batch_size == 0:
                    self.meta_optimizer.step()
                    self.meta_optimizer.zero_grad()

    # in this function, we will collect trajectories in a interactive manner.
    # In the original paper of MAML, the trajectories used for updating adaptive model and meta model are different
    def interactivate_fit(self, meta_task, controller):
        # [TODO]
        pass

    def adapt(self):
        if self.adapt_data is None:
            print('No adapt data!')
            return 0

        normllized_adapt_data = self.normalize(self.adapt_data, self.data_mu, self.data_sigma)
        normllized_adapt_label = self.normalize(self.adapt_label, self.label_mu, self.label_sigma)

        # copy the parameter from meta model, avoiding mess the meta model up
        self.adapt_model.copy(self.meta_model)
        self.adapt_model.train()
        adapt_optimizer = torch.optim.Adam(self.meta_model.params(), lr=self.adapt_lr)
        
        # new data should be added with data_process function
        # fine-tune
        for i in range(self.adapt_iter):
            adapt_optimizer.zero_grad()
            loss = self.inner_fit(self.adapt_model,normllized_adapt_data, normllized_adapt_label, create_graph=False)
            adapt_optimizer.step()
        return loss
        
    # this function should be called after the adapt function
    def predict(self, s, a):
        s = CUDA(torch.tensor(s).float())
        a = CUDA(torch.tensor(a).float())
        inputs = torch.cat((s, a), axis=1)
        inputs = self.normalize(inputs, self.data_mu, self.data_sigma)
        self.adapt_model.eval()
        with torch.no_grad():
            delta_state = self.adapt_model(inputs)
            delta_state = self.denormalize(delta_state, self.label_mu, self.label_sigma)
            delta_state = delta_state.cpu().detach().numpy()
        return delta_state

    def test(self, s, a, x_g):
        s = CUDA(torch.tensor(s).float())
        a = CUDA(torch.tensor(a).float())
        inputs = torch.cat((s, a), axis=1)
        inputs = self.normalize(inputs, self.data_mu, self.data_sigma)
        self.adapt_model.eval()
        with torch.no_grad():
            delta_state = self.adapt_model(inputs)
            delta_state = self.denormalize(delta_state, self.label_mu, self.label_sigma)
            delta_state = delta_state.cpu().detach().numpy()

        mse_error = np.sum((delta_state-x_g)**2)
        return delta_state, mse_error

    def reset_adapt_data(self):
        self.adapt_data = None
        self.adapt_label = None

    def reset_meta_data(self):
        self.meta_data = None
        self.meta_label = None

    def finish_one_episode(self):
        # save one episode data
        if self.meta_data is not None:
            self.meta_data_collector.append(self.meta_data)
            self.meta_label_collector.append(self.meta_label)

        # prepare for next adaption
        self.reset_adapt_data()
        self.reset_meta_data()
