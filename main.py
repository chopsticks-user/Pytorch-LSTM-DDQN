import numpy as np
import gym
import torch as T
from torch.optim import lr_scheduler as ls
import lin_qn, lstm_qn, conv_qn, environment, tools
#from ultis import plot_learning_curve

if __name__ == "__main__":
    # set to false after the first training section to continue training
    new_traning = True

    if new_traning:
        agent = tools.new_agent(in_features = [250, 160, 3], out_features = 6, hidden_lstm_size = 16, n_hidden_lstm_layers = 1, 
                                hidden_lin_size = 16, replay_size = 3, bidirectional = True, file_path = "Load/AirRaid")

        tools.training_phase(agent, gym.make("ALE/AirRaid-v5", render_mode = "human"))
    else:
        agent = tools.train_load("Load/AirRaid")
        tools.training_phase(agent, gym.make("ALE/AirRaid-v5", render_mode = "human"))
