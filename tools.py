import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import namedtuple, deque

transition_values = namedtuple("transition_values", ("current_state", "action", "next_state", "reward", "done"))
