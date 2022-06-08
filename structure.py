from unicodedata import bidirectional
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as O
from collections import deque
import os

#contains n_lstm_layers lstm layers and 2 fc layers
class DuelLSTM(nn.Module):
    def __init__(self, in_features = 1, out_features = 1, hidden_lstm_size = 1, n_lstm_layers = 1, 
                hidden_lin_size = 16, alpha = 0.001, bidirectional = False, dropout = 0.0, 
                file_path = None):
        super().__init__()
        #64 is the optimal size of each hidden layer
        self.device = T.device("cuda")
        in_features = int(np.prod(in_features))
        self.hidden_lin_size = hidden_lin_size
        self.hidden_lstm_size = hidden_lstm_size
        self.lstm = nn.LSTM(input_size = in_features, hidden_size = hidden_lstm_size, num_layers = n_lstm_layers, 
                            dropout = 0.0, bidirectional = bidirectional)
        if bidirectional:
            hidden_lstm_size *= 2
        #compare performance to non-additional lin_layer model
        #two independent streams of duel q net

        
        #self.fc0 = nn.Linear(in_features, hidden_lin_size)
        #self.fc1 = nn.Linear(hidden_lin_size, hidden_lstm_size)
        

        #advantage stream
        self.fc_advantage = nn.Linear(hidden_lstm_size, hidden_lin_size)
        self.final_fc_advantage = nn.Linear(hidden_lin_size, out_features)

        #value stream
        self.fc_value = nn.Linear(hidden_lstm_size, hidden_lin_size)
        self.final_fc_value = nn.Linear(hidden_lin_size, out_features)

        #compare to HuberLoss
        self.loss = nn.HuberLoss()
        self.optimizer = O.Adam(self.parameters(), alpha, betas = (0.9, 0.9999))
        self.to(self.device)

    def forward(self, x):
        #reshape x to 3D tensor to be accepted as an lstm input
        #tanh activation function included in lstm layers
        #Error: input.size(-1) must be equal to input_size. Expected 2, got 128
        #torch lstm requires 3D tensors as inputs
        #Error: "addmm_cuda" not implemented for 'Byte' -> done by adding ".float()" before
        #passing to the forward function

        
        x, _ = self.lstm(x.to(self.device))
        x = T.squeeze(x)
        

        #using view or reshape messes up obs matrices
        #use transpose or permute instead
        
        
        #x = F.relu(self.fc0(x))
        #x = F.relu(self.fc1(x))
        

        value = F.leaky_relu(self.fc_value(x))
        value = self.final_fc_value(value)
                  

        advantage = F.leaky_relu(self.fc_advantage(x))
        advantage = self.final_fc_advantage(advantage)

        q = F.log_softmax(value + advantage - advantage.mean(), 1)

        return q

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#contains all fully connected layers
class DuelLinear(nn.Module):
    def __init__(self, in_features = 1, out_features = 1, hidden_layers_dims = [1, 1], 
                    learning_rate = 0.001, file_path = None):
        super().__init__()
        #self.device = "cpu"
        self.device = T.device("cuda")
        #learning rate??
        #use named_tuple -> done
        #exception when hidden_layers_sizes contains any element with a value of 0
        #hidden layers are hidden from named_parameters call -> done
        #be mindful of NaN values
        #still using cpu -> done
        #might need to move first_layer and final_layer in to a list contains all layers -> done
        #might not need in/out_features_dim (but out_features needs to be 
        #converted back to its orginial dimension)
        #self.to(self.device) might not be needed
        #edit Adam optim -> done

        if file_path:
            self.load(file_path)
            return

        self.lr = learning_rate
        in_features = int(np.prod(in_features))

        hidden_layers_dims[:0] = [in_features]
        hidden_layers_dims[len(hidden_layers_dims):] = [out_features]
        self.module = nn.ModuleList([nn.Linear(hidden_layers_dims[i], hidden_layers_dims[i + 1]).to(device = self.device)
                        for i in range(len(hidden_layers_dims) - 1)])

        self.loss = nn.HuberLoss()
        self.optimizer = O.Adam(self.parameters(), lr = self.lr)
        self.to(self.device)
        return

    #exception when hidden_layers_sizes contains any element with a value of 0
    #if passed sample has input dimenson != in_features, return dimension error 
    #(may caused by not initilzing in_features)
    def forward(self, x):
        for i in range(len(self.module) - 1):
            x = F.relu(self.module[i](x.to(self.device)))
        x = self.module[-1](x)
        return x

    def attributes(self):
        for p in self.named_parameters():
            print(p)

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
class DuelConv(nn.Module):
    def __init__(self):
        super().__init__()
        pass


