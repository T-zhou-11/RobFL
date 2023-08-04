import numpy as np
import torch
import torch.nn as nn
import copy

from sklearn.mixture import BayesianGaussianMixture
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA

class CenterLoss(nn.Module):

    def __init__(self, num_classes=100, feat_dim=1024, init_centers=torch.randn(100, 1024), use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(init_centers).cuda()#100 x feats- for 100 centers
        else:
            self.centers = nn.Parameter(init_centers)

    def get_centers(self):
        return self.centers.clone().detach()

    def update(self, mean_center):
        self.centers = nn.Parameter(mean_center).cuda()

    def update_(self, x, labels, sample):
        batch_size = x.size(0)

        for i in range(self.num_classes):
            fea =  torch.Tensor([x[index] for index in range(batch_size) if labels[index]==i])
            center_i = (1/sample) *torch.cumsum(fea, dim=0)
            self.init_cneter[i] += center_i

    def forward(self,x, labels):
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())  # 1 * dist + -2 * x * centers.t()
        # x2+centers2-2x*centers = (x-centers)2
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes).cuda()
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        sep_dist = []
        center_dist = []
        for i in range(batch_size):

            k= mask[i].clone().to(dtype=torch.int8)
            k= -1* k +1
            kk= k.clone().to(dtype=torch.bool) #True/False
            sep_value = distmat[i][kk]
            sep_value = sep_value.clamp(min=1e-12, max=1e+12) # for numerical stability
            sep_dist.append(sep_value)

            center_value = distmat[i][mask[i]]
            center_value = center_value.clamp(min=1e-12, max=1e+12)
            center_dist.append(center_value)
        sep_dist = torch.cat(sep_dist)

        sep_loss = sep_dist.mean()

        center_dist = torch.cat(center_dist)
        center_loss = center_dist.mean()

        loss = center_loss - 0.001* sep_loss

        return loss


def mean_fea_center(model, data, label, num_classes, fea_dim):
    party_set = TensorDataset(data, label)
    party_loader = DataLoader(party_set, batch_size=256, shuffle=True, num_workers=4)
    model.eval()
    centers = torch.zeros(num_classes, fea_dim).cuda()
    num_per_cls = [0 for _ in range(num_classes)]
    for batch_idx, (inputs, targets) in enumerate(party_loader):
        inputs = inputs.cuda()
        batch_size = inputs.size(0)
        fea, _ = model(inputs)

        for index in range(batch_size):
            label = targets[index].item()
            num_per_cls[label] += 1
            centers[label] += fea[index]
    for c in range(num_classes):
        centers[c] = centers[c] / float(num_per_cls[c])
    return centers


def center_filter(centers, k):
    party_centers = []
    for i in range(len(centers)):
        party_centers.append(centers[i].flatten().cpu().numpy())
    input_centers = np.array(party_centers)

    print('## shape of input centers: ', input_centers.shape)
    mean = np.mean(input_centers, axis=0)
    margin = np.linalg.norm(input_centers - mean, ord=None, axis=1, keepdims=False)
    benign = margin.argsort()[:k]
    return benign

def gmm_center_filter(centers):
    party_centers = []
    for i in range(len(centers)):
        party_centers.append(centers[i].flatten().cpu().numpy())
    input_centers = np.array(party_centers)

    pca = PCA(n_components=10)
    dim_result = pca.fit_transform(input_centers)
    dim_fea = torch.tensor(dim_result)
    gmm = BayesianGaussianMixture(n_components=2, max_iter=120, tol=1e-3, reg_covar=0, weight_concentration_prior_type='dirichlet_process')
    gmm.fit(dim_fea)
    prob = gmm.predict_proba(dim_fea)
    prob = np.argmax(prob, axis=1).tolist()
    maxlabel = max(prob, key=prob.count)
    benign = [i for i, x in enumerate(prob) if x is maxlabel]
    return benign


def median_grad_filter(lgrads):
    party_grads = []
    for i in range(len(lgrads)):
        norm = torch.norm(lgrads[i], p='fro')
        party_grads.append(norm)
    m = median(party_grads)
    outlier = [ i for i in range(len(party_grads)) if party_grads[i]> (5*m)]
    benign = [i for i in range(len(party_grads)) if i not in outlier]
    return benign

def median(grad):
    list = copy.deepcopy(grad)
    list.sort()
    list_length = len(list)
    if list_length % 2 == 0:
        return (list[int(list_length / 2) - 1] + list[int(list_length / 2)]) / 2
    return list[int(list_length / 2)]
