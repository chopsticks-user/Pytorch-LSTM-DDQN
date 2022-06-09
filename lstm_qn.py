from unicodedata import bidirectional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as O
from torch.optim import lr_scheduler as ls
from collections import namedtuple, deque

from structure import DuelLSTM
from memory import Memory

transition_values = namedtuple("transition_values", ("current_state", "action", "next_state", "reward", "terminal_state"))

class DuelQNet(nn.Module):
    def __init__(self, in_features = 1, out_features = 1, hidden_lstm_size = 1, n_hidden_lstm_layers = 1, 
                hidden_lin_size = 32, gamma = 0.9, alpha = 0.01, epsilon = 1.0, epsilon_min = 0.01, 
                epsilon_decay = 5000, memory_capacity = 50000, batch_size = 128, replay_size = 3, 
                bidirectional = False, dropout = 0.0, environment = None, file_path = None):
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
        self.memory = Memory(capacity = memory_capacity, replay_size = replay_size)
        self.replay_size = replay_size
        self.batch_size  = batch_size

        self.critic_net = DuelLSTM(in_features = in_features, out_features = out_features, hidden_lstm_size = hidden_lstm_size, 
                                    n_lstm_layers = n_hidden_lstm_layers, hidden_lin_size = hidden_lin_size, alpha = alpha,
                                    bidirectional = bidirectional, dropout = dropout)
        
        #evaluation
        self.total_reward = 0
        self.last_reward = 0
        #self.epoch_reward = deque(maxlen = 10000)
        self.total_loss = 0.0
        self.average_loss = 0.0
        self.loss = 0.0
        self.current_step = 0
        self.last_action = 0

        self.lr_scheduler = ls.ReduceLROnPlateau(self.critic_net.optimizer, "min", 0.9, 1)

        self.to(self.device)

    def lr_decay_method(self, method = None):
        self.lr_scheduler = method

    def execute_lr_decay(self):
        self.lr_scheduler.step()

    #only call this function if using ReduceOnPlateau Scheduler
    def execute_lr_decay_on_plateau(self):
        self.lr_scheduler.step(self.loss)

    def transition(self, current_state, action, next_state, reward, terminal_state):
        if self.current_step % 3:
            if reward != 0:
                self.last_reward += reward
            self.current_step += 1
            return
        #set terminal_state = 1.0 or 0.0 for calculating target net
        self.current_step += 1
        reward += self.last_reward
        self.last_reward = 0
        terminal_state = 1.0 if terminal_state == False else 0.0
        #add epoch reward
        self.total_reward += reward
        self.memory.update(current_state, action, next_state, reward, terminal_state)

        #update evalution instances

    #independent from obs if trasition was called before (obs has been updated to the memory)
    #def decision(self, obs):
    def decision(self):
        self.epsilon -= self.epsilon_step if self.epsilon > self.epsilon_end else 0
        if np.random.rand() > self.epsilon:
            #passing the most recent replay batch (slow performance, might be fixed using dp)
            #self.momory.data[-1] is the last replay batch
            #from n current state replay values to a 3D tensor of shape (1, n, in_features)
            #~ (batch_size, seq_length, in_features) -> input (batch_first = True) 
            #of LSTM-net forward function

            if self.current_step % 3:
                return self.last_action

            current_state_replay_batch = T.tensor(np.array(transition_values(*zip(*self.memory.replay)).current_state), dtype = T.float32).to(self.device)

            #torch.stack requires 1D-list inputs
            #current_state_replay_batch = T.stack(current_state_replay_batch)

            #draw back: at the i-th sence in a replay batch of size n (i in range(0, n)), 
            #only i + 1 sences selected => LSTM net perform best when i = n - 1
            #-> no solution so far, cannot wait until all the sences have been collected
            #as the next actions might be selected randomly -> the whole replay batch will be skipped
            #fixed length replay batch (each step, using n sences in a batch) even after
            #terminal state could be a solution
            #-> done, using another memory implementation

            actions = self.critic_net.forward(T.flatten(current_state_replay_batch, start_dim = 1))

            #each sequence has its own output, also means 
            #when there are than one scence in the current replay batch,
            #the forward function return more than one set of actions 
            #->done, by selecting an action in the last set

            #assume that actions has 2 dimensions
            #self.last_action = action.item() % actions.size(dim = 1)

            action = T.argmax(actions[-1]).item()
            self.last_action = action
            return self.last_action
        return np.random.choice(self.action_space)

    def learn(self):
        #wait until batch_size replay batches have been completed
        if self.current_step / 3 < self.batch_size:
            return

        #indices variable is used for debugging only
        indices, batch = self.memory.sample(self.batch_size)
        
        current_state_batch = T.tensor(np.array([*batch.current_state]), dtype = T.float32).to(self.device)
        action_batch = T.tensor(np.array([*batch.action]), dtype = T.long).to(self.device)
        next_state_batch = T.tensor(np.array([*batch.next_state]), dtype = T.float32).to(self.device)
        reward_batch = T.tensor(np.array([*batch.reward]), dtype = T.float32).to(self.device)
        terminal_state_batch = T.tensor(np.array([*batch.terminal_state]), dtype = T.float32).to(self.device)
        

        '''
        current_state_batch = T.tensor(np.array(transition_values(*zip(*minibatch)).current_state)).to(self.device)
        action_batch = T.tensor(np.array(transition_values(*zip(*minibatch)).action)).to(self.device)
        next_state_batch = T.tensor(np.array(transition_values(*zip(*minibatch)).next_state)).to(self.device)
        reward_batch = T.tensor(np.array(transition_values(*zip(*minibatch)).reward)).to(self.device)
        '''

        #cast action_batch to type long before using it as indices
        #action_batch = action_batch.long()

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

        #have not found a prebuild function
        temp = self.critic_net.forward(T.unsqueeze(T.flatten(current_state_batch, start_dim = 1), 0)).to(self.device)
        temp = T.squeeze(temp)
        policy = T.zeros([self.batch_size], dtype = T.float32).to(self.device)
        for i in range(self.batch_size):
                policy[i] = temp[i, action_batch[i]]

        max_q = self.critic_net.forward(T.unsqueeze(T.flatten(next_state_batch, start_dim = 1), 0)).to(self.device)
        max_q = T.squeeze(max_q)
        max_q = T.max(max_q, 1).values
        target = reward_batch + T.mul(self.gamma, max_q * terminal_state_batch)
        #target = T.mul(self.gamma, max_q)

        #cast target to float to prevent error when calling loss.backward
        #Error: found dtype Double but expected float -> done
        loss = (self.critic_net.loss(policy, target))

        
        #L2 regularization
        L2_lambda = 0.01
        L2_norm = sum(p.pow(2.0).sum() for p in self.parameters())
        loss += L2_lambda * L2_norm
        

        self.critic_net.optimizer.zero_grad()
        loss.backward()
        self.critic_net.optimizer.step()
        self.loss = loss

        # a leaf Variable that requires grad is being used in an in-place operation. -> done by
        # using not-in-place operation
        self.total_loss = self.total_loss + loss
        self.average_loss = self.total_loss / (self.current_step - self.batch_size * 3)
        self.alpha = self.critic_net.optimizer.param_groups[0]["lr"]
        #self.lr_scheduler.step(loss)
        
        #self.alpha = self.lr_scheduler.get_last_lr()

    
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        