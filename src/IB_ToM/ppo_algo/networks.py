import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.out)

    def forward(self, x):
       x = self.tanh(self.fc1(x))
       x = self.tanh(self.fc2(x))
       logits = self.out(x)
       probs = F.softmax(logits, dim=-1)
       return probs

class ToMActor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, tom_input_size, tom_hidden_size):
        super(ToMActor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.tom_fc = nn.Linear(tom_input_size, tom_hidden_size)
        self.out = nn.Linear(hidden_size + tom_hidden_size, output_size)
        self.tanh = nn.Tanh()
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.out)
        orthogonal_init(self.tom_fc)

    def forward(self, x, x_tom):
       x_tom = self.tanh(self.tom_fc(x_tom))
       x = self.tanh(self.fc1(x))
       x = self.tanh(self.fc2(x))
       logits = self.out(torch.concat([x, x_tom], dim=-1))
       probs = F.softmax(logits, dim=-1)
       return probs

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.out)

    def forward(self, x):
      x = self.tanh(self.fc1(x))
      x = self.tanh(self.fc2(x))
      value = self.out(x)
      return value

class ConActor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ConActor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mu_layer = nn.Linear(hidden_size, output_size)
        self.log_std_layer = nn.Parameter(torch.zeros(1, output_size))
        orthogonal_init(self.fc1)
        # orthogonal_init(self.fc2)
        orthogonal_init(self.mu_layer, gain=0.01)


    def forward(self, x):
       x = F.tanh(self.fc1(x))
       # x = F.relu(self.fc2(x))
       mu = self.mu_layer(x)
       std = torch.exp(self.log_std_layer)
       return mu, std

    def get_action(self, x):
        mu, std = self.forward(x)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        action = torch.clamp(action, -1, 1)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.cpu().numpy(), log_prob.detach().cpu().numpy()

