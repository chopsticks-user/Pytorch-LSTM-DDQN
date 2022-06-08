import random as R
from collections import namedtuple, deque
from tools import transition_values
import torch as T
import numpy as np

#seperate replay batches
class Memory(object):
    def __init__(self, capacity = 10000, replay_size = 5):
        self.capacity = capacity
        self.replay_size = replay_size
        self.memory_count = 0
        self.data = deque([], maxlen = capacity)
        self.replay = deque([], maxlen = replay_size)

    def update(self, *args):
        self.data.append(args)
        self.replay.append(args)
        self.memory_count += 1

    def sample(self, batch_size):
        indices = R.sample(range(self.memory_count if self.memory_count <= self.capacity else self.capacity), batch_size)
        batch = transition_values(*zip(*map(list(self.data).__getitem__, indices)))
        return indices, batch

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#each cell is a replay batch 
#reduce duplicate "memory" names -> done
class ReplayMemory(object):
    def __init__(self, capacity = 10000, replay_size = 5):
        #memory contains capacity/replay_size replay batchs
        self.capacity = capacity
        self.replay_size = replay_size
        self.data = deque([], maxlen = self.capacity)
        #memory-overflow check
        self.current_step = 0

        #when the first replay batch is created, this value increase by 1
        self.n_completed_replay_batch = -1

        #scence index in a replay batch (0 -> n - 1 if replay_size = n)
        self.scence_index = 0

        #end-of-replay-batch check
        self.next_replay_batch = True

    def update(self, *args):
        input = transition_values(*args)
        if self.next_replay_batch:
            if self.n_completed_replay_batch >= self.capacity:
                #pop the first replay batch if memory is full
                self.n_completed_replay_batch -= 2
                self.data.popleft()
            self.n_completed_replay_batch += 1
            self.scence_index = 0
            self.next_replay_batch = False
            self.data.append(deque([], maxlen = self.replay_size))
        else:
            self.scence_index += 1
        self.data[-1].append(input)

        #if the state is terminal (terminal_state = 0.0)
        #if self.scence_index == self.replay_size - 1 or input.terminal_state == 0.0:
            #self.next_replay_batch = True

        #ignore terminal states, all replay batches have a fixed length of n
        if self.scence_index == self.replay_size - 1:
            self.next_replay_batch = True

    def sample(self, batch_size):
        #sample a deque takes O(m*n) time while sampling a list takes O(n) time
        #ignore the current replay batch since it is incompleted
        indices = R.sample(range(self.n_completed_replay_batch), batch_size)
        batch = map(list(self.data).__getitem__, indices)
        
        #batch = T.tensor(list(batch))
        #cannot return batch as a tensor, 
        #ValueError: expected sequence of length 5 at dim 1 (got 1)
        #-> done, by doing the following steps
        #1, batch = [*batch] -> a list of values of the unzipped batch
        #2, batch = np.asarray(batch) -> assign the list to a np array (as ~ no copy)
        
        batch = [*batch]
        batch = np.asarray(batch)
        
        #current_state_batch = np.array(batch[:,:,0], dtype = np.float32)
        #print(batch[:,:,1].dtype)

        #print(current_state_batch)

        #when using T.as_tensor, get error: "not a sequence"

        #return indices, current state batch, action batch, next state batch, 
        #reward batch, terminal state batch 
        return indices, batch[:, :, 0], batch[:, :, 1], batch[:, :, 2], batch[:, :, 3], batch[:, :, 4]

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''