from collections import deque
import numpy as np
from collections import namedtuple

Experience = namedtuple("Experience",
    field_names=["state", "action", "reward", "next_state", "done", "pi"])

class Memory():
    def __init__(self, max_size=1000):
        self.size = max_size  # maximum size of buffer
        self.memory = []  # internal memory (list)
        self.idx = 0

    def add(self, state, action, reward, next_state, done, pi):
        e = Experience(state, action, reward, next_state, done, pi)
        if len(self.memory) < self.size:
            self.memory.append(e)
        else:
            self.memory[self.idx] = e
            self.idx = (self.idx + 1) % self.size

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.memory)),
                               size=batch_size,
                               replace=False)
        return [self.memory[ii].state for ii in idx], [self.memory[ii].action for ii in idx], \
               [self.memory[ii].reward for ii in idx], [self.memory[ii].next_state for ii in idx], \
               [self.memory[ii].done for ii in idx]

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

