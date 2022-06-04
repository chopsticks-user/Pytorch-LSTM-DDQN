import random as R
from collections import namedtuple, deque

#reduce duplicate "memory" names -> done
class DequeMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = deque([], maxlen = self.capacity)
        self.cur_mem_p = 0

    def update(self, *args):
        if self.cur_mem_p < self.capacity:
            self.data.append(*args)
            self.cur_mem_p += 1
            return
        self.data.popleft()
        self.data.appendleft(*args)

    def sample(self, batch_size):
        return R.sample(range(batch_size), batch_size), R.sample(self.data, batch_size)

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
class ListMemory(object):
    def __init__(self, capacity = 100000):
        self.capacity = capacity
        self.data = [()] * capacity
        self.current = 0
    
    def store(self, *args):
        #terminal state excluded
        self.data[self.current] = args
        self.current += 1

    def sample(self, size):
        #zip error
        indices, minibatch = zip(*(R.sample(list(enumerate(self.data[:self.current + 1])), size)))
        return indices, minibatch