
import bisect
from config.constants import QObjectIndices

class Priority_Q:
    def __init__(self):
        self.Q = []

    def add(self, item):        
        # list is sorted in descending order, so next action is retrieved in O(1) by pop(0)
        index = bisect.bisect_left(self.Q, -1*item[QObjectIndices.ACTIONTIME], key= lambda x : -1*x[QObjectIndices.ACTIONTIME])
        self.Q.insert(index, item)

    def pop(self):
        if self.is_empty():
            return None
        return self.Q.pop(0)

    def is_empty(self):
        return len(self.Q) == 0