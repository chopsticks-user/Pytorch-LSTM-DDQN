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
from lstm_qn import DuelQNet
#from memory import Memory

transition_values = namedtuple("transition_values", ("current_state", "action", "next_state", "reward", "terminal_state"))

def train_save(agent):
    T.save(agent, agent.file_path)
    print(f"Training saved at {agent.file_path}.")

def train_load(file_path):
    train_agent = T.load(file_path)
    train_agent.train()
    print(f"Training loaded at {file_path}.")
    return train_agent

def inference_load(file_path):
    # with T.no_grad():
    train_agent = T.load(file_path)
    train_agent.eval()
    print(f"Testing loaded at {file_path}.")
    return train_agent

def new_agent(in_features = 1, out_features = 1, hidden_lstm_size = 1, n_hidden_lstm_layers = 1, 
                hidden_lin_size = 32, gamma = 0.9, alpha = 0.01, epsilon = 1.0, epsilon_min = 0.01, 
                epsilon_decay = 5000, memory_capacity = 50000, batch_size = 128, replay_size = 3, 
                bidirectional = False, dropout = 0.0, environment = None, file_path = None):

    agent = DuelQNet(in_features = in_features, out_features = out_features, hidden_lstm_size = hidden_lstm_size, 
                    n_hidden_lstm_layers = n_hidden_lstm_layers, hidden_lin_size = hidden_lin_size, gamma = gamma, 
                    alpha = alpha, epsilon = epsilon, epsilon_min = epsilon_min, epsilon_decay = epsilon_decay, 
                    memory_capacity = memory_capacity, batch_size = batch_size, replay_size = replay_size, 
                    bidirectional = bidirectional, dropout = dropout, environment = environment, file_path = file_path)
                    
    agent.lr_decay_method(ls.ExponentialLR(agent.critic_net.optimizer, 0.9999))
    return agent

def training_phase(agent, environment):
    environment.reset()
    for i in range(1000):
        obs = environment.reset()
        done = False
        agent.reward = 0
        actions = []
        action = 0

        while not done:
            action = agent.decision()
            actions.append(action)
            new_obs, reward, done, _ =  environment.step(action)
            agent.transition(obs, action, new_obs, reward, done)
            obs = new_obs
            agent.learn()
            agent.execute_lr_decay()
            
            if(agent.current_step % 10 == 0):
                print("Episode {}, Step {}: Action: {}, Total reward: {}., Epsilon: {:.4f}, Loss: {}"
                .format(i, agent.current_step, actions, agent.total_reward, agent.epsilon, agent.loss), 
                ", Learning rate: ", agent.alpha)
                print(f"Average Loss: {agent.average_loss}, ")
                actions.clear()

            #environment.render()  
        
        train_save(agent)

    environment.close()

#def testing_phase():