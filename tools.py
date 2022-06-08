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
#from memory import Memory

transition_values = namedtuple("transition_values", ("current_state", "action", "next_state", "reward", "terminal_state"))

