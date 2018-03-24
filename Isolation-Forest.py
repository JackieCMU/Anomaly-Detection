# 2018/3/22 Jiaqi He
# Isolation Forest Implementation

import random
import math
import numpy as np

class Node:

    def __init__(self, attr = None, value = None, left = None, right = None, size = None):
        self.split_attr = attr
        self.split_val = value
        self.left = left
        self.right = right
        self.size = size


class isolationForest:

    def __init__(self, data, sub_sample_size = 256, forest_size = 100, maximum_depth = 8):
        self.attributes = data.shape[1]
        self.sub_sample_size = sub_sample_size
        self.forest_size = forest_size
        self.forest = []

        for i in range(self.forest_size):
            sub_sample = np.asarray(random.sample(data.tolist(), self.sub_sample_size))
            root =  Node()
            self.forest.append(self.iTree(root, sub_sample, 0, maximum_depth))

    def get_forest(self):
        return self.forest

    def __len__(self):
        return len(self.forest)

    def __iter__(self):
        return _isolation_iterator(self.forest)

    def iTree(self, root, sub_sample, e, l):

        if e >= l or sub_sample.shape[0] <= 1:
            return Node(size = sub_sample.shape[0])

        split_axis = [i for i,val in enumerate(sub_sample.T) if max(val)>min(val)]
        random_axis =  split_axis[random.randint(0, len(split_axis)-1)]

        lower, upper = min(sub_data[:,random_axis]), max(sub_data[:, random_axis])
        random_split = random.uniform(lower, upper)

        sample_lower = sub_sample[sub_sample[:,random_axis] < random_split]
        sample_upper = sub_sample[sub_sample[:,random_axis] >= random_split]
        root = Node(random_axis, random_split, size = sub_sample.shape[0])
        root.left = self.iTree(root.left, sample_lower, e + 1, l)
        root.right = self.iTree(root.right, sample_upper, e + 1, l)

        return root

    def get_path_len(self, root, sample, h_lim):
        e = 0
        while root.split_val:
           e += 1
           if e >= h_lim:
               break
           if sample[root.split_axis] < root.split_val:
               root = root.left
           else:
               root = root.right
        return e, root

    def evaluate_forest(self, sample, h_lim):
        path_len = []
        for point in sample:
            res = []
            for root in self:
                e, root = self.get_path_len(root, point, h_lim)
                res.append(e + c(root.size))
            path_len.append(res)
        return cal_anomaly(path_len, self.sub_sample_size)

def H(x):
    return math.log(x) + 0.5772156649

def c(x):
    if x> 2:
        path_length = 2 * H(x - 1) - 2 * (x - 1) / x
    elif x == 2:
        path_length = 1
    else:
        path_length = 0
    return path_length

def cal_anomaly(length, sub_sample_size):
    return [2**(-np.mean(val)/c(sub_sample_size)) for val in length]
