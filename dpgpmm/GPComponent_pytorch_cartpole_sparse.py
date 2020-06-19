
import random
import numpy as np
from loguru import logger

import torch
import gpytorch
from gpytorch.constraints import GreaterThan, Positive, LessThan


def CUDA(var):
    #return var
    return var.cuda() if torch.cuda.is_available() else var


class ExactGPR(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, input_dim, params):
        super(ExactGPR, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=input_dim, 
                lengthscale_constraint=LessThan(params[0])
            ),
            outputscale_constraint=GreaterThan(params[1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def reset_parameters(self, params):
        self.likelihood.noise_covar.initialize(noise=params[0])
        self.mean_module.initialize(constant=params[1])
        self.covar_module.initialize(outputscale=params[2])
        self.covar_module.base_kernel.initialize(lengthscale=params[3:8])


class SparseGPR(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, input_dim, params):
        super(SparseGPR, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=input_dim, 
                lengthscale_constraint=LessThan(params[0])
            ),
            outputscale_constraint=GreaterThan(params[1])
        )
        # use some training data to initialize the inducing_module
        if train_x is None:
            train_x = CUDA(torch.zeros((1, input_dim)))
        self.inducing_module = gpytorch.kernels.InducingPointKernel(self.covar_module, inducing_points=train_x, likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.inducing_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def reset_parameters(self, params):
        self.likelihood.noise_covar.initialize(noise=params[0])
        self.mean_module.initialize(constant=params[1])
        self.covar_module.initialize(outputscale=params[2])
        self.covar_module.base_kernel.initialize(lengthscale=params[3:8])


class GPComponent(object):
    def __init__(self, test_data, args):
        self.n = 0
        self.data = None
        self.index_list = []
        
        self.lr = args.lr
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.input_dim = self.state_dim + self.action_dim
        self.gp_iter = args.gp_iter

        # parameters for sparse GP
        self.max_inducing_point = args.max_inducing_point
        self.trigger_induce = args.trigger_induce

        # prior of the kernel parameters
        self.param = [
            1e-3,   # noise_covar initilize and constraint
            0.0,    # constant initilize
            0.1,    # outputscale initilize
            1.0, 1.0, 1.0, 1.0, 1.0, # [lengthscale initilize]
            100.0,  # lengthscale_constraint
            0.0001   # outputscale_constraint
        ] 
        self.param = CUDA(torch.tensor(self.param))

        # initialize model and likelihood
        model_list = []
        likelihood_list = []
        for m_i in range(self.state_dim):
            likelihood = CUDA(gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(self.param[0])))
            model = CUDA(ExactGPR(None, None, likelihood, self.input_dim, self.param[8:10]))
            model.reset_parameters(self.param)
            likelihood_list.append(model.likelihood)
            model_list.append(model)
        
        # initialize model list
        self.model = gpytorch.models.IndependentModelList(*model_list)
        self.likelihood = gpytorch.likelihoods.LikelihoodList(*likelihood_list)

        # initialize optimizer
        self.optimizer = torch.optim.Adam([{'params': self.model.parameters()}], lr=self.lr)
        self.mll = gpytorch.mlls.SumMarginalLogLikelihood(self.likelihood, self.model)

        ##################################################################################################################
        # initialize a sparse model
        sparse_model_list = []
        sparse_likelihood_list = []
        for m_i in range(self.state_dim):
            likelihood = CUDA(gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(self.param[0])))
            model = CUDA(SparseGPR(None, None, likelihood, self.input_dim, self.param[8:10]))
            model.reset_parameters(self.param)
            sparse_likelihood_list.append(model.likelihood)
            sparse_model_list.append(model)
        
        self.sparse_model = gpytorch.models.IndependentModelList(*sparse_model_list)
        self.sparse_likelihood = gpytorch.likelihoods.LikelihoodList(*sparse_likelihood_list)

        # initialize optimizer
        self.sparse_optimizer = torch.optim.Adam([{'params': self.sparse_model.parameters()}], lr=self.lr)
        self.sparse_mll = gpytorch.mlls.SumMarginalLogLikelihood(self.sparse_likelihood, self.sparse_model)

        # the sparse model will never be used for prdiction, the flag should always be train
        self.sparse_model.train()
        self.sparse_likelihood.train()
        ##################################################################################################################

    def reset_parameters(self):
        for m_i in range(self.state_dim):
            self.model.models[m_i].reset_parameters(self.param)
        
    def reset_point_for_test(self, test_data):
        self.reset_parameters()

    def train_sparse_model(self):
        # reset training data and parameters
        # DO NOT reset parameters, use the parameters of last time
        for m_i in range(self.state_dim):
            # prepare data
            train_x = self.data[m_i][:, :self.input_dim]
            train_y = self.data[m_i][:, self.input_dim]

            # use random index
            index = random.sample(range(0, self.n), self.max_inducing_point)
            inducing_points = train_x[index, :]
            
            # use the new data points as inducing points, so satrt from the end
            #i_start = self.n - self.max_inducing_point
            #inducing_points = train_x[i_start:, :]

            # use the old point
            #inducing_points = train_x[:self.max_inducing_point, :]

            self.sparse_model.models[m_i].set_train_data(train_x, train_y, strict=False)
            self.sparse_model.models[m_i].inducing_module.inducing_points = CUDA(torch.nn.Parameter(inducing_points))
            # [A FEATURE OF GPYTORCH]:
            # GPytorch does not support dynamically changing inducing_points, because two variables are cached
            # these two variables need to be deleted then they will be re-calculated.
            if hasattr(self.sparse_model.models[m_i], "_cached_kernel_mat"):
                delattr(self.sparse_model.models[m_i].inducing_module, '_cached_kernel_mat')
            if hasattr(self.sparse_model.models[m_i], "_cached_kernel_inv_root"):
                delattr(self.sparse_model.models[m_i].inducing_module, '_cached_kernel_inv_root')

        # training stage
        self.sparse_gp_iter = 3
        for i in range(self.sparse_gp_iter):
            self.sparse_optimizer.zero_grad()
            output_func = self.sparse_model(*self.sparse_model.train_inputs)
            loss = -self.sparse_mll(output_func, self.sparse_model.train_targets)
            loss.backward()
            self.sparse_optimizer.step()

        # calulate the prediction y for all inducing points
        # NOTE: here we should use non-sparse model to do the inference rather than the sparse one.
        test_inducing_point = []
        for m_i in range(self.state_dim):
            test_inducing_point.append(self.sparse_model.models[m_i].inducing_module.inducing_points.detach())
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            #sample_func = self.likelihood(*self.model(*test_inducing_point))
            sample_func = self.model(*test_inducing_point)
        
        # reset data to the inducing points
        for m_i in range(self.state_dim):
            self.data[m_i] = self.data[m_i][:self.max_inducing_point] # reset the length
            self.data[m_i][:, :self.input_dim] = test_inducing_point[m_i]
            self.data[m_i][:, self.input_dim] = sample_func[m_i].mean

    def train_model(self):
        # check if we need to run the sparse process
        # check sparse condition first and modify the data
        if self.n > self.trigger_induce:
            self.train_sparse_model()
            # since the data has been truncated, a new length should be calculated
            self.n = len(self.data[0])
            nonsparse_gp_iter = 2*self.gp_iter
        else:
            nonsparse_gp_iter = self.gp_iter

        # reset training data and parameters
        # DO NOT reset parameters, use the parameters of last time
        for m_i in range(self.state_dim):
            # prepare data
            train_x = self.data[m_i][:, :self.input_dim]
            train_y = self.data[m_i][:, self.input_dim]
            self.model.models[m_i].set_train_data(train_x, train_y, strict=False)

        # training stage
        self.model.train()
        self.likelihood.train()
        for i in range(nonsparse_gp_iter):
            self.optimizer.zero_grad()
            output_func = self.model(*self.model.train_inputs)
            loss = -self.mll(output_func, self.model.train_targets)
            loss.backward()
            self.optimizer.step()

    def get_data_number(self):
        return len(self.data[0])

    def get_point(self):
        data_x = self.data[0][:, :self.input_dim].cpu().numpy()
        data_y = []
        for d_i in range(self.state_dim):
            data_y.append(self.data[d_i][:, self.input_dim:self.input_dim+1].cpu().numpy())
        data = np.concatenate((data_x, *data_y), axis=1)
        return data

    def merge_point(self, new_tensor_data, new_list):
        # the data to be merged is expected to be a torch.Tensor
        for d_i in range(self.state_dim):
            self.data[d_i] = torch.cat((self.data[d_i], new_tensor_data[d_i]), dim=0)

        self.index_list += new_list
        self.n += len(new_list)

    def add_point(self, x, i):
        data_x = CUDA(torch.Tensor(x[:, :self.input_dim]))
        data_y = CUDA(torch.Tensor(x[:, self.input_dim:]))
        self.index_list.append(i)
        self.n += 1

        if self.data is None:
            self.data = []
            # the data to be merged is expected to be a torch.Tensor
            for d_i in range(self.state_dim):
                new_data = torch.cat((data_x, data_y[:, d_i:d_i+1]), dim=1)
                self.data.append(new_data)
        else:
            # the data to be merged is expected to be a torch.Tensor
            for d_i in range(self.state_dim):
                new_data = torch.cat((data_x, data_y[:, d_i:d_i+1]), dim=1)
                self.data[d_i] = torch.cat((self.data[d_i], new_data), dim=0)

    def log_posterior_pdf(self, x):
        # prepare data
        x = CUDA(torch.Tensor(x))
        test_x = x[:, :self.input_dim]
        test_y = x[:, self.input_dim:]

        # get the log likelihood
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            #sample_func = self.likelihood(*self.model(test_x, test_x, test_x, test_x))
            sample_func = self.model(test_x, test_x, test_x, test_x)
            log_ppf = 0
            for f_i in range(len(sample_func)):
                log_ppf += sample_func[f_i].log_prob(test_y[:, f_i]).item()
        # since we added four likelihood
        return log_ppf/len(sample_func)

    def predict(self, x):
        sample_func = self.predict_distribution(x)

        ds = []
        for i in range(self.state_dim):
            ds.append(sample_func[i].loc.cpu().numpy())
        ds = np.array(ds).T
        return ds

    # this is used for KL divergence calculation
    # this part does not need to normalize the output
    def predict_distribution(self, x):
        # prepare data
        test_x = CUDA(torch.Tensor(x[:, :self.input_dim]))

        self.model.eval()
        self.likelihood.eval()
        # get the log likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_all = test_x.repeat(self.state_dim, 1, 1)
            sample_func = self.model(*test_all)
            #sample_func_lik = self.likelihood(*sample_func)
        return sample_func
