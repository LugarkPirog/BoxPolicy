from collections import deque
from random import sample


class BaseReplayBuffer:

    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)

    def add(self, *obs):
        self.buffer.extend(zip(*obs))

    def sample(self, n):
        return sample(self.buffer, n)

    def __len__(self):
        return len(self.buffer)


class ReplayBuffer(BaseReplayBuffer):
    pass


#class HER(BaseReplayBuffer):
#    raise NotImplementedError

