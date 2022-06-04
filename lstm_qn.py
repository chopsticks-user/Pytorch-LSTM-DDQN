from attr import validate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as O
from torch.optim import lr_scheduler as ls
from collections import namedtuple

from structure import DuelLSTM
from memory import DequeMemory
from tools import transition_values

class DuelQNet(nn.Module):
    def __init__(self, in_features = 1, out_features = 1, hidden_lstm_size = 1, n_hidden_lstm_layers = 1, 
                hidden_lin_size = 256, gamma = 0.9, alpha = 0.001, epsilon = 1.0, epsilon_min = 0.01, 
                epsilon_decay = 5000, memory_capacity = 100000, batch_size = 64, environment = None, file_path = None):
        super().__init__()
        self.device = T.device("cuda")
        self.env = environment
        self.file_path = file_path

        #learning variables
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_end = epsilon_min
        self.epsilon_step = (epsilon - epsilon_min) / epsilon_decay

        #assume that n_actions = out_features
        self.action_space = [i for i in range(out_features)]

        #memory and batch
        self.memory_size = memory_capacity
        self.memory = DequeMemory(memory_capacity)
        self.batch_size  = batch_size

        self.critic_net = DuelLSTM(in_features = in_features, out_features = out_features, hidden_lstm_size = hidden_lstm_size, 
                                    n_lstm_layers = n_hidden_lstm_layers, hidden_lin_size = hidden_lin_size, alpha = alpha)
        #evaluation
        self.total_reward = 0
        self.loss = 0

        self.lr_scheduler = ls.ReduceLROnPlateau(self.critic_net.optimizer, "min", 0.9, 1)

        self.to(self.device)

    def transition(self, *args):
        #terminal state excluded
        trans_vals = transition_values(*args)
        self.memory.update(trans_vals)
        self.total_reward += trans_vals.reward

    def decision(self, obs):
        self.epsilon -= self.epsilon_step if self.epsilon > self.epsilon_end else 0
        if np.random.rand() > self.epsilon:
            actions = self.critic_net.forward(T.flatten(T.tensor(obs)).float()) #+ self.critic_net2.forward(T.flatten(T.tensor(obs)))
            return T.argmax(actions).item()
        return np.random.choice(self.action_space)

    def learn(self):
        if self.memory.cur_mem_p < self.batch_size:
            return

        indices, minibatch = self.memory.sample(self.batch_size)

        current_state_batch = T.tensor(np.array(transition_values(*zip(*minibatch)).current_state)).to(self.device)
        action_batch = T.tensor(np.array(transition_values(*zip(*minibatch)).action)).to(self.device)
        next_state_batch = T.tensor(np.array(transition_values(*zip(*minibatch)).next_state)).to(self.device)
        reward_batch = T.tensor(np.array(transition_values(*zip(*minibatch)).reward)).to(self.device)

        #cast type to long to be indices
        action_batch = action_batch.long()

        '''
        policy1 = (self.critic_net.forward(current_state_batch)[:, indices, action_batch]).to(self.device)
        policy2 = (self.critic_net2.forward(current_state_batch)[:, indices, action_batch]).to(self.device)
        next_est_q1 = (self.critic_net.forward(next_state_batch)[:, indices, action_batch]).to(self.device)
        next_est_q2 = (self.critic_net2.forward(next_state_batch)[:, indices, action_batch]).to(self.device)
        q_est_min = T.min(next_est_q1, next_est_q2).to(self.device)
        target = reward_batch + (T.mul(self.gamma, q_est_min))

        loss1 = self.critic_net.loss(target, policy1).to(self.device)
        loss2 = self.critic_net2.loss(target, policy2).to(self.device)
        loss_min = T.min(loss1, loss2)
        self.critic_net.optimizer.zero_grad()
        self.critic_net2.optimizer.zero_grad()
        loss_min.backward()
        self.critic_net.optimizer.step()
        self.critic_net2.optimizer.step()

        self.lr_scheduler.step()
        self.alpha = self.lr_scheduler.get_last_lr()
        self.lr_scheduler2.step()
        self.alpha = self.lr_scheduler2.get_last_lr()

        self.loss = loss_min
        '''
        #cast type to float to prevent error when forwarding
        policy = (self.critic_net.forward(current_state_batch.float())[:, indices, action_batch]).to(self.device)
        #max_q = self.critic_net.forward(next_state_batch).to(self.device)
        max_q = T.max(self.critic_net.forward(next_state_batch.float()), 2).values
        target = reward_batch + (T.mul(self.gamma, max_q))

        #cast target to float to prevent error when calling loss.backward
        #Error: found dtype Double but expected float -> done
        loss = (self.critic_net.loss(policy, target.float()))
        self.critic_net.optimizer.zero_grad()
        loss.backward()
        self.critic_net.optimizer.step()
        self.loss = loss
        self.alpha = self.critic_net.optimizer.param_groups[0]["lr"]
        #self.lr_scheduler.step(loss)
        
        #self.alpha = self.lr_scheduler.get_last_lr()

    def lr_decay_method(self, method = None):
        self.lr_scheduler = method

    def execute_lr_decay(self):
        self.lr_scheduler.step()

    #only call this function if using ReduceOnPlateau Scheduler
    def execute_lr_decay_on_plateau(self):
        self.lr_scheduler.step(self.loss)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        