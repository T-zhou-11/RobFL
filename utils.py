import torch
from torch import nn
import numpy as np
from collections import Counter

def pgd(model, data, target, criterion, epsilon, step_size, num_steps, category='Madry',rand_init=True):
    model.eval()

    if category == "trades":
        x_adv = data.detach() + 0.01 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
        _, nat_logit = model(data)
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for k in range(num_steps):
        x_adv.requires_grad_()
        _, output = model(x_adv)
        model.zero_grad()
        with torch.enable_grad():
            loss_adv = criterion(output, target)
            #print("K %s, fealoss %s, total loss %s"%(k,fea_loss, loss_adv ))
        loss_adv.backward(retain_graph=True)
        eta = step_size * x_adv.grad.sign()
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv


def get_cls_num_list(each_party_label, dataset, num_parties):
    cls_num_list = [[] for _ in range(num_parties)]
    cls_sum = []
    num_class = 100 if dataset == "cifar100" else 10
    for i in range(num_parties):
        cls_num_list[i] = Counter(each_party_label[i])
        cls_sum.append(len(each_party_label[i]))
    return cls_num_list,cls_sum
