import sys

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
import torch.autograd as autograd

is_cuda = torch.cuda.is_available()
# is_cuda = False
torch.backends.cudnn.enabled = False


def MSELoss(input, target):
    return torch.sum((input - target) ** 2) / input.data.nelement()


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class Policy(nn.Module):

    def __init__(self, hidden_size, num_inputs, action_space):
        super(Policy, self).__init__()
        self.action_space = action_space
        num_outputs = action_space
        # self.bn0 = nn.BatchNorm1d(num_inputs)
        # self.bn0.weight.data.fill_(1)
        # self.bn0.bias.data.fill_(0)

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        # self.emedbing=
        self.lstm = nn.LSTM(input_size=num_inputs, hidden_size=hidden_size)
        self.hidden_dim = hidden_size
        # self.bn1 = nn.BatchNorm1d(hidden_size)
        # self.bn1.weight.data.fill_(1)
        # self.bn1.bias.data.fill_(0)
        #
        # self.linear2 = nn.Linear(hidden_size, hidden_size)
        # self.bn2 = nn.BatchNorm1d(hidden_size)
        # self.bn2.weight.data.fill_(1)
        # self.bn2.bias.data.fill_(0)

        self.V = nn.Linear(hidden_size, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

        self.mu = nn.Linear(hidden_size, num_outputs)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

        self.L = nn.Linear(hidden_size, num_outputs ** 2)
        self.L.weight.data.mul_(0.1)
        self.L.bias.data.mul_(0.1)

        self.tril_mask = Variable(torch.tril(torch.ones(
            num_outputs, num_outputs), diagonal=-1).unsqueeze(0))
        self.diag_mask = Variable(torch.diag(torch.diag(
            torch.ones(num_outputs, num_outputs))).unsqueeze(0))
        if is_cuda:
            self.tril_mask = self.tril_mask.cuda()
            self.diag_mask = self.diag_mask.cuda()
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if is_cuda:
            return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim).cuda()),
                    autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).cuda())
        else:
            return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                    autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, inputs):
        x, u = inputs
        # x = self.bn0(x)
        # x, self.hidden = self.lstm(x, self.hidden)
        x, _ = self.lstm(x)
        x = torch.tanh(x)
        # x = F.tanh(self.linear2(x))
        V = self.V(x)
        mu = torch.tanh(self.mu(x))

        Q = None
        if u is not None:
            num_outputs = mu.size(2)
            L = self.L(x).view(-1, num_outputs, num_outputs)
            L = L * self.tril_mask.expand_as(L) + torch.exp(L) * self.diag_mask.expand_as(L)
            P = torch.bmm(L, L.transpose(2, 1))

            u_mu = u - mu
            A = -0.5 * torch.bmm(torch.bmm(u_mu, P), u_mu.transpose(2, 1))[:, :, 0]

            Q = A + V

        return mu, Q, V


class NAF:

    def __init__(self, gamma, tau, hidden_size, num_inputs, action_space):
        self.action_space = action_space
        self.num_inputs = num_inputs

        self.model = Policy(hidden_size, num_inputs, action_space)
        self.target_model = Policy(hidden_size, num_inputs, action_space)
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)

        self.gamma = gamma
        self.tau = tau
        if is_cuda:
            self.model = self.model.cuda()
            self.target_model = self.target_model.cuda()
            # self.optimizer = self.optimizer.cuda()

        hard_update(self.target_model, self.model)

    def select_action(self, state, action_noise=None, param_noise=None):
        self.model.eval()
        # print(Variable(state))
        if is_cuda:
            V_s = Variable(state).cuda()
            ac_noise = torch.Tensor(action_noise.noise()).cuda()
        else:
            V_s = Variable(state)
            ac_noise = torch.Tensor(action_noise.noise())
        mu, _, _ = self.model((V_s, None))
        self.model.train()
        mu = mu.data
        if action_noise is not None:
            mu += ac_noise
        if is_cuda:
            mu = mu.cpu()
        return mu.clamp(-1, 1)

    def update_parameters(self, batch):
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        mask_batch = Variable(torch.cat(batch.mask))
        next_state_batch = Variable(torch.cat(batch.next_state))
        if is_cuda:
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            mask_batch = mask_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        _, _, next_state_values = self.target_model((next_state_batch, None))

        reward_batch = reward_batch.unsqueeze(1)
        mask_batch = mask_batch.unsqueeze(1)
        expected_state_action_values = reward_batch + (self.gamma * mask_batch + next_state_values)

        _, state_action_values, _ = self.model((state_batch, action_batch))

        loss = MSELoss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        # loss.backward(retain_graph=True)
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm(self.model.parameters(), 1)
        self.optimizer.step()

        soft_update(self.target_model, self.model, self.tau)

        return loss.item(), 0

    def save_model(self, env_name, suffix="", model_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if model_path is None:
            model_path = "models/naf_{}_{}".format(env_name, suffix)
        print('Saving model to {}'.format(actor_path))
        torch.save(self.model.state_dict(), model_path)

    def load_model(self, model_path):
        print('Loading model from {}'.format(model_path))
        self.model.load_state_dict(torch.load(model_path))
