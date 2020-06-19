import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np

'''
According to deepmind imp and that[*] implementation
'''





'''
fundamental building blocks MLP with n (hidden layers + Relu )   # Comment, the sequential infor is neglected
one final layer without activation
**Pytorch will automatically operate on the last dim**
Structure: 
input(B, seq_len, input_dim) 
    V linear_1 [W_1 \in (input_dim, hidden_size_1)]
 variable (B, seq_len, hidden_size_1)
    V Relu
 variable (B, seq_len, hidden_size_1)
    V linear_2 [W_2 \in (hidden_size_1, hidden_size_2)]
 varaible (B, seq_len, hidden_size_2)
    V Relu
 variable (B, seq_len, hidden_size_2)
    V ...
   ...
    V last linear without relu
 variable (B, seq_len, output_size)
'''
class MLP(nn.Module):
    '''
    Apply MLP to the final axis of a 3D tensor
    '''
    def __init__(self, input_size, output_size_list):
        super().__init__()
        self.input_size = input_size
        self.output_size_list = output_size_list    # e.g. [128, 128, 128, 128]
        network_size_list = [input_size] + self.output_size_list # e.g. [2, 128, 128, 128, 128]
        network_list = []
        # iteratively build
        for i in range(1, len(network_size_list) - 1):
            network_list.append(nn.Linear(network_size_list[i-1], network_size_list[i], bias=False))
            network_list.append(nn.ReLU())
        network_list.append(nn.Linear(network_size_list[-2], network_size_list[-1]))
        self.mlp = nn.Sequential(*network_list)

    def forward(self, x):
        self.batch_size, self.seq_len, self.input_dim = x.size()
        return self.mlp(x)


########################################
# NOTICE: modules of ANP
#   Latent Encoder
#   Deterministic Encoder
#   Decoder
########################################

# NOTICE: LatentEncoder
'''
Latent Encoder (With self-attention)
Structure:
input context_x(batch, seq_len, input_d), context_y(batch, seq_len, output_d)

1. concate
input = concate (x, y)  (batch, seq_len, all_d)


2. pass final axis through MLP
latent = encoder_mlp(input)
(b, len, latent_dim) = mlp((b, len, all_d))

3. Optional self-attention


4. mean
latent_s_c = latent.mean(dim=1) 
latent_s_c (b, 1, latent_dim)

5. reparameterization trick

'''
class LatentEncoder(nn.Module):
    def __init__(
        self,
        input_x_dim,
        input_y_dim,
        hidden_dim_list,    # the dims of hidden starts of mlps
        latent_dim=32,      # the dim of last axis of sc and z..
        self_attention_type="dot",
        use_self_attn=True,
        attention_layers=2,
        use_lstm=False
    ):
        super().__init__()
        self.input_dim = input_x_dim + input_y_dim
        self.hidden_dim_list = hidden_dim_list
        self.hidden_dim = hidden_dim_list[-1]
        self.latent_dim = latent_dim

        if latent_dim != hidden_dim_list[-1]:
            print('Warning, Check the dim of latent z and the dim of mlp last layer!')

        # NOTICE: On my paper, we seems to substitute the mlp with LSTM
        #  but we actually add a LSTM before the mlp
        if use_lstm:
            # self._encoder = LSTMBlock(input_dim, hidden_dim, batchnorm=batchnorm, dropout=dropout,
            #                           num_layers=n_encoder_layers)
            pass
        else:
            self.latent_encoder_mlp = MLP(input_size=self.input_dim, output_size_list=hidden_dim_list)
            # Output should be (b, seq_len, hidden_dim_list[-1])

        if use_self_attn:
            self._self_attention = Attention(
                self.latent_dim,
                self_attention_type,
                attention_layers,
                rep="identity",
            )
            pass
        self.penultimate_hidden_num = int(0.5*(self.hidden_dim + self.latent_dim))
        # print('0.5*(self.hidden_dim + self.latent_dim) =', 0.5*(self.hidden_dim + self.latent_dim))
        self.penultimate_layer = nn.Linear(self.hidden_dim, self.penultimate_hidden_num)

        self.mean_layer = nn.Linear(self.penultimate_hidden_num, self.latent_dim)
        self.std_layer = nn.Linear(self.penultimate_hidden_num, self.latent_dim)
        self._use_lstm = use_lstm
        self._use_self_attn = use_self_attn

    def forward(self, x, y):
        # print('x.size() =', x.size())
        # print('y.size() =', y.size())
        encoder_input = torch.cat([x, y], dim=-1)
        # encoder_input (b, seq_len, input_dim=input_x_dim + input_y_dim)
        # = (b, seq_len, input_dim)

        # NOTICE:Pass final axis through MLP
        # print('encoder_input.size() =', encoder_input.size())
        hidden = self.latent_encoder_mlp(encoder_input)
        # hidden (b, seq_len, hidden_dim)


        # NOTICE: Aggregator: take the mean over all points
        if self._use_self_attn:
            hidden_s_i = self._self_attention(hidden, hidden, hidden)
            hidden_s_c = hidden_s_i.mean(dim=1)
        else:
            hidden_s_i = hidden
            hidden_s_c = hidden_s_i.mean(dim=1)
        # hidden_s_c (b, 1, hidden_dim)
        # Comment, Here assume all the sequence (x, y pair) comes
        # from the same stochastic process (dynamics)
        # pytorch will squeeze automatically, which is actually not desired
        # need to be unsqueezed later

        # NOTICE: Have further MLP layers that map to the parameters of the Gaussian latent
        #  MLP[d, 2d] on the paper
        #   First apply intermediate relu layer
        hidden_z = torch.relu(self.penultimate_layer(hidden_s_c))
        # hidden_z (b, 1, 0.5(hidden_dim + latent_dim))
        # = (b, 1, latent_dim), when $hidden_dim=latent_dim$

        # NOTICE: Then apply further linear layers to output latent mu and log sigma
        mu = self.mean_layer(hidden_z)
        # mu (b, 1, latent_dim)
        log_sigma = self.std_layer(hidden_z)
        # log_sigma (b, 1, latent_dim)

        # Compute sigma
        sigma = 0.1 + 0.9 * torch.sigmoid(log_sigma)

        return torch.distributions.Normal(mu, sigma), mu, sigma
        #
        # mu (b, latent_dim)
        # sigma (b, latent_dim)
        # need to be unsqueezed later


# NOTICE: LatentEncoder
'''
DeterministicEncoder Encoder (With self-attention)
Can be attended with cross-attention *Whether to put the cross-attention in this encoder?
'''
class DeterministicEncoder(nn.Module):
    def __init__(
        self,
        input_x_dim,
        input_y_dim,
        hidden_dim_list,  # the dims of hidden starts of mlps
        latent_dim=32,  # the dim of last axis of r..
        self_attention_type="dot",
        use_self_attn=True,
        attention_layers=2,
        use_lstm=False,
        cross_attention_type="dot",
        attention_dropout=0,
    ):
        super().__init__()
        self.input_dim = input_x_dim + input_y_dim
        self.hidden_dim_list = hidden_dim_list
        self.hidden_dim = hidden_dim_list[-1]
        self.latent_dim = latent_dim
        self.use_self_attn = use_self_attn

        if latent_dim != hidden_dim_list[-1]:
            print('Warning, Check the dim of latent z and the dim of mlp last layer!')

        # NOTICE: In my paper, we seems to substitute the mlp with LSTM
        #  but we actually add a LSTM before the mlp
        if use_lstm:
            # self._encoder = LSTMBlock(input_dim, hidden_dim, batchnorm=batchnorm, dropout=dropout,
            #                           num_layers=n_encoder_layers)
            pass
        else:
            self.deter_encoder_mlp = MLP(input_size=self.input_dim, output_size_list=hidden_dim_list)
            # Output should be (b, seq_len, hidden_dim_list[-1])
            # output: (b, seq_len, hidden_dim)

        if use_self_attn:
            self._self_attention = Attention(
                self.latent_dim,
                self_attention_type,
                attention_layers,
                rep="identity",
            )
            pass
        self._cross_attention = Attention(
            self.latent_dim,
            cross_attention_type,
            x_dim=input_x_dim,
            attention_layers=attention_layers,
        )


    def forward(self, context_x, context_y, target_x):
        # Concatenate x and y along the filter axes
        encoder_input = torch.cat([context_x, context_y], dim=-1)
        # encoder_input (b, seq_len, input_dim=input_x_dim + input_y_dim)
        # = (b, seq_len, input_dim)

        # Pass final axis through MLP
        hidden_r_i = self.deter_encoder_mlp(encoder_input)
        # hidden_r_i (b, seq_len, latent_dim)

        if self.use_self_attn:
            hidden_r_i = self._self_attention(hidden_r_i, hidden_r_i, hidden_r_i)
        else:
            hidden_r_i = hidden_r_i

        # Apply attention as mean aggregation
        # In the ANP paper, context_x and target_x are first passed by a mlp
        # In my paper, all x are first passed by lstm
        #
        h = self._cross_attention(context_x, hidden_r_i, target_x)
        # context_x     (b, seq_len, input_x_dim)           # Key
        # hidden_r_i    (b, seq_len, latent_dim)            # Value
        # target_x      (b, target_seq_len, input_x_dim)    # Query
        #
        return h        #(b, target_seq_len, latent_dim)


# NOTICE: Decoder
'''
Decoder
1. concatenate the target_x and latent variables r_star and z
2. Then pass them input a MLP
3. According to deepmind imp, then split the hidden to get mu and sigma
    Maybe using reparamerization trick will break something here
 
Can be attended with cross-attention *Whether to put the cross-attention in this encoder?

From the deepmind implementation
decoder_output_sizes = [HIDDEN_SIZE]*2 + [2] => decoder_hidden_dim_list[-1] = 2
Here 2 comes from y_dim * 2

The operation on latent variables should be completed outside the decoder
'''
class Decoder(nn.Module):
    def __init__(
            self,
            x_dim,
            y_dim,
            mid_hidden_dim_list,  # the dims of hidden starts of mlps
            latent_dim=32,  # the dim of last axis of sc and z..
            use_deterministic_path=True,  # whether use d_path or not will change the size of input
            use_lstm=False,
        ):
        super(Decoder, self).__init__()

        self.hidden_dim_list = mid_hidden_dim_list + [y_dim*2]

        if use_deterministic_path:
            self.decoder_input_dim = 2 * latent_dim + x_dim
        else:
            self.decoder_input_dim = latent_dim + x_dim

        if use_lstm:
            # self._decoder = LSTMBlock(hidden_dim_2, hidden_dim_2, batchnorm=batchnorm, dropout=dropout,
            #                           num_layers=n_decoder_layers)
            pass
        else:
            # self._decoder = BatchMLP(hidden_dim_2, hidden_dim_2, batchnorm=batchnorm, dropout=dropout,
            #                          num_layers=n_decoder_layers)
            self.decoder_mlp = MLP(input_size=self.decoder_input_dim, output_size_list=self.hidden_dim_list)
        # self._mean = nn.Linear(hidden_dim_2, y_dim)
        # self._std = nn.Linear(hidden_dim_2, y_dim)
        self._use_deterministic_path = use_deterministic_path
        # self._min_std = min_std
        # self._use_lvar = use_lvar

    def forward(self, r, z, target_x):
        # r:        (b, target_seq_len, latent_dim)
        # z:        (b, target_seq_len, latent_dim)
        # target_x: (b, target_seq_len, x_dim)

        # concatenate target_x and representation
        if self._use_deterministic_path:
            z = torch.cat([r, z], dim=-1)
            # z (b, target_seq_len, 2 * latent_dim )
        hidden_mu_sigma = torch.cat([z, target_x], dim=-1)
        # (b, target_len, 2 * latent_dim + x_dim)

        mu_sigma = self.decoder_mlp(hidden_mu_sigma)
        # (b, target_len, 2 * y_dim)

        # Get the mean and the variance ???
        # print('type(mu_sigma) =', type(mu_sigma))
        # print('mu_sigma.size() =', mu_sigma.size())
        # output_debug = mu_sigma.split(chunks=2, dim=-1)
        # print('output_debug[0].size() =', output_debug[0].size())

        mu, log_sigma = mu_sigma.chunk(chunks=2, dim=-1)
        # print('mu.size() =', mu.size())
        # print('log_sigma.size() =', log_sigma.size())
        # mu (b, target_len, y_dim)
        # sigma (b, target_len. y_dim)

        # Bound the variance
        sigma =0.1 + 0.9 * F.softplus(log_sigma)

        # Get the distibution
        dist = torch.distributions.Normal(mu, sigma)
        return dist, mu, sigma


# NOTICE: Attention
'''
Attention

'''
class AttnLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        torch.nn.init.normal_(self.linear.weight, std=in_channels ** -0.5)

    def forward(self, x):
        x = self.linear(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        hidden_dim,
        attention_type,
        attention_layers=2,
        n_heads=8,
        x_dim=1,
        rep="mlp",
        dropout=0,
        batchnorm=False,
    ):
        super().__init__()
        self._rep = rep

        if self._rep == "mlp":
            self.batch_mlp_k = BatchMLP(
                x_dim,
                hidden_dim,
                attention_layers,
                dropout=dropout,
                batchnorm=batchnorm,
            )
            self.batch_mlp_q = BatchMLP(
                x_dim,
                hidden_dim,
                attention_layers,
                dropout=dropout,
                batchnorm=batchnorm,
            )

        if attention_type == "uniform":
            self._attention_func = self._uniform_attention
        elif attention_type == "laplace":
            self._attention_func = self._laplace_attention
        elif attention_type == "dot":
            self._attention_func = self._dot_attention
        elif attention_type == "multihead":
            self._W_k = nn.ModuleList(
                [AttnLinear(hidden_dim, hidden_dim) for _ in range(n_heads)]
            )
            self._W_v = nn.ModuleList(
                [AttnLinear(hidden_dim, hidden_dim) for _ in range(n_heads)]
            )
            self._W_q = nn.ModuleList(
                [AttnLinear(hidden_dim, hidden_dim) for _ in range(n_heads)]
            )
            self._W = AttnLinear(n_heads * hidden_dim, hidden_dim)
            self._attention_func = self._multihead_attention
            self.n_heads = n_heads
        elif attention_type == "ptmultihead":
            self._W = torch.nn.MultiheadAttention(
                hidden_dim, n_heads, bias=False, dropout=dropout
            )
            self._attention_func = self._pytorch_multihead_attention
        else:
            raise NotImplementedError

    def forward(self, k, v, q):
        if self._rep == "mlp":
            k = self.batch_mlp_k(k)
            q = self.batch_mlp_q(q)
        rep = self._attention_func(k, v, q)
        return rep

    def _uniform_attention(self, k, v, q):
        total_points = q.shape[1]
        rep = torch.mean(v, dim=1, keepdim=True)
        rep = rep.repeat(1, total_points, 1)
        return rep

    def _laplace_attention(self, k, v, q, scale=0.5):
        k_ = k.unsqueeze(1)
        v_ = v.unsqueeze(2)
        unnorm_weights = torch.abs((k_ - v_) * scale)
        unnorm_weights = unnorm_weights.sum(dim=-1)
        weights = torch.softmax(unnorm_weights, dim=-1)
        rep = torch.einsum("bik,bkj->bij", weights, v)
        return rep

    def _dot_attention(self, k, v, q):
        scale = q.shape[-1] ** 0.5
        unnorm_weights = torch.einsum("bjk,bik->bij", k, q) / scale
        weights = torch.softmax(unnorm_weights, dim=-1)

        rep = torch.einsum("bik,bkj->bij", weights, v)
        return rep

    def _multihead_attention(self, k, v, q):
        outs = []
        for i in range(self.n_heads):
            k_ = self._W_k[i](k)
            v_ = self._W_v[i](v)
            q_ = self._W_q[i](q)
            out = self._dot_attention(k_, v_, q_)
            outs.append(out)
        outs = torch.stack(outs, dim=-1)
        outs = outs.view(outs.shape[0], outs.shape[1], -1)
        rep = self._W(outs)
        return rep

    def _pytorch_multihead_attention(self, k, v, q):
        # Pytorch multiheaded attention takes inputs if diff order and permutation
        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)
        o = self._W(q, k, v)[0]
        return o.permute(1, 0, 2)


class NPBlockRelu2d(nn.Module):
    """Block for Neural Processes."""

    def __init__(
        self, in_channels, out_channels, dropout=0, batchnorm=False, bias=False
    ):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout)
        self.norm = nn.BatchNorm2d(out_channels) if batchnorm else False

    def forward(self, x):
        # x.shape is (Batch, Sequence, Channels)
        # We pass a linear over it which operates on the Channels
        x = self.act(self.linear(x))

        # Now we want to apply batchnorm and dropout to the channels. So we put it in shape
        # (Batch, Channels, Sequence, None) so we can use Dropout2d & BatchNorm2d
        x = x.permute(0, 2, 1)[:, :, :, None]

        if self.norm:
            x = self.norm(x)

        x = self.dropout(x)
        return x[:, :, :, 0].permute(0, 2, 1)


class BatchMLP(nn.Module):
    """Apply MLP to the final axis of a 3D tensor (reusing already defined MLPs).
    Args:
        input: input tensor of shape [B,n,d_in].
        output_sizes: An iterable containing the output sizes of the MLP as defined
            in `basic.Linear`.
    Returns:
        tensor of shape [B,n,d_out] where d_out=output_size
    """

    def __init__(
        self, input_size, output_size, num_layers=2, dropout=0, batchnorm=False
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.initial = NPBlockRelu2d(
            input_size, output_size, dropout=dropout, batchnorm=batchnorm
        )
        self.encoder = nn.Sequential(
            *[
                NPBlockRelu2d(
                    output_size, output_size, dropout=dropout, batchnorm=batchnorm
                )
                for _ in range(num_layers - 2)
            ]
        )
        self.final = nn.Linear(output_size, output_size)

    def forward(self, x):
        x = self.initial(x)
        x = self.encoder(x)
        return self.final(x)



if __name__ == '__main__':
    torch.manual_seed(0)

    input_size = 2
    output_list = [4, 4, 4, 4]
    x = torch.randn(3, 5, 3)
    y = torch.randn(3, 5, 2)
    print('x.size() =', x.size())

    encoder_test = LatentEncoder(input_x_dim=2, input_y_dim=3, hidden_dim_list=[4,4,4], latent_dim=4)

    dist, mu, sigma = encoder_test(x, y)
    print('dist =', dist)
    print('mu = ', mu)
    print('sigma = ', sigma)

    # with torch.no_grad():
    #     mlp_test = MLP(input_size=input_size, output_size_list=output_list)
    #     print('mlp_test.mlp =', mlp_test.mlp)
    #     my_output = mlp_test(x)
    #
    #
    #     mlp_tt = BatchMLP(input_size=input_size, output_size=4, num_layers=4)
    #     print('mlp_tt.initial =', mlp_tt.initial)
    #     print('mlp_tt.encoder =', mlp_tt.encoder)
    #     print('mlp_tt.final =', mlp_tt.final)
    #
    #     tt_output = mlp_tt(x)
    #     print('my_output.size() =', my_output)
    #     print('tt_output.size() =', tt_output)