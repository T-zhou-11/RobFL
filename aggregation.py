import torch
import copy
import numpy as np
from copy import deepcopy
import time
from sklearn.metrics import roc_auc_score
import torch
import byzantine

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def fedavg(param_list, data_sizes):
    """
    Based on the description in https://arxiv.org/abs/1602.05629
    grads: list of gradients.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    device: computation device.
    data_size: amount of training data of each worker device
    """

    n = len(param_list)
    # compute global model update
    global_update = torch.zeros(param_list[0].size()).cuda()
    for i, grad in enumerate(param_list):
        global_update += grad
    global_update /= n
    return global_update

def krum(param_list, f):
    """
    Based on the description in https://papers.nips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html
    gradients: list of gradients.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    device: computation device.
    """

    n = len(param_list)

    # compute pairwise Euclidean distance
    dist = torch.zeros((n, n)).cuda()
    for i in range(n):
        for j in range(i + 1, n):
            d = torch.norm(param_list[i] - param_list[j])
            dist[i, j], dist[j, i] = d, d

    # sort distances and get model with smallest sum of distances to closest n-f-2 models
    sorted_dist, _ = torch.sort(dist, dim=-1)
    global_update = param_list[torch.argmin(torch.sum(sorted_dist[:, 0:(n - f - 1)], dim=-1))]
    return global_update


def trim_mean(param_list, f):
    """
    Based on the description in https://arxiv.org/abs/1803.01498
    gradients: list of gradients.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    device: computation device.
    """

    n = len(param_list)

    # trim f biggest and smallest values of gradients
    sorted, _ = torch.sort(torch.cat(param_list, dim=1), dim=-1)
    global_update = torch.mean(sorted[:, f:(n - f)], dim=-1)

    return global_update


def median(param_list, f):
    """
    Based on the description in https://arxiv.org/abs/1803.01498
    gradients: list of gradients.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    device: computation device.
    """

    # compute median of gradients
    global_update, _ = torch.median(torch.cat(param_list, dim=1), dim=-1)

    return global_update



def score(gradient, v, f):
    num_neighbours = v.shape[1] - 2 - f
    sorted_distance = torch.square(v - gradient).sum(axis=0).sort()
    return torch.sum(sorted_distance[1:(1+num_neighbours)]).asscalar()

def nearest_distance(gradient, c_p):
    sorted_distance = torch.square(c_p - gradient).sum(axis=1).sort(axis=0)
    return sorted_distance[1].asscalar()