from __future__ import print_function

# import mxnet as mx
# from mxnet import nd, autograd, gluon
import copy
import time
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
import random
import argparse

import byzantine
import aggregation
# import nd_aggregation
from center import CenterLoss, mean_fea_center, center_filter, gmm_center_filter,gmm_grad_filter,median_grad_filter
from utils import pgd, get_cls_num_list
from nets.cnn import CNNMnist

# os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
# np.warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset", default='mnist', type=str)
    parser.add_argument("--datapath", help="path to dataset", default='./data', type=str)
    parser.add_argument("--bias", help="degree of non-IID to assign data to workers", type=float, default=0.5)
    parser.add_argument("--net", help="net", default='cnn', type=str, choices=['mlr', 'cnn', 'fcnn'])
    parser.add_argument("--batch_size", help="batch size", default=256, type=int)
    parser.add_argument("--local_bs", default=512, type=int)
    parser.add_argument("--lr", help="learning rate", default=0.1, type=float)
    parser.add_argument("--num_parties", help="# participants", default=50, type=int)
    parser.add_argument("--num_epochs", help="# epochs", default=1000, type=int)
    parser.add_argument("--local_iter", help="# local epochs", default=1, type=int)
    parser.add_argument("--gpu", help="index of gpu", default=5, type=int)
    parser.add_argument("--seed", help="seed", default=2024, type=int)
    parser.add_argument("--num_attackers", help="# attackers", default=16, type=int)
    parser.add_argument("--byz_type", help="type of attack", default='dba', type=str)
    parser.add_argument("--aggregation", help="aggregation rule", default='avg', type=str)
    parser.add_argument("--sep_frac", default=1e-4, type=float)
    parser.add_argument("--warmup", default=0, type=int)
    parser.add_argument("--sep_train", default=False, action='store_true')
    parser.add_argument("--detect", default=False, action='store_true')
    parser.add_argument("--detmode", default='gmm', type=str)
    parser.add_argument("--filter", default=False, action='store_true')
    parser.add_argument("--filmode", help="reweight or remove", default='remove', type=str)
    parser.add_argument("--scalemode", default='mode0', type=str)
    parser.add_argument("--save", default='mnist', type=str)
    parser.add_argument('--schedule_lr', type=int, nargs='+', default=[600],
                        help='Decrease learning rate at these epochs.')
    return parser.parse_args()


def adjust_lr(lr, epoch, schedule_lr):
    if epoch in schedule_lr:
        lr = 0.1* lr
    return lr

def evaluate_accuracy(data_iterator, net):
    correct = 0
    for i, (data, label) in enumerate(data_iterator):
        data, label = data.cuda(), label.cuda()
        _, output = net(data)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(label.view_as(pred)).sum().item()
    test_accuracy = correct / len(data_iterator.dataset) * 100.
    return test_accuracy

def evaluate_pgd10(data_iterator, net):
    correct = 0
    for i, (data, label) in enumerate(data_iterator):
        data, label = data.cuda(), label.cuda()
        inputs_adv = pgd(net, data, label, nn.CrossEntropyLoss(), 25.5 / 255, 2 / 255, 10)
        data, label = inputs_adv.cuda(), label.cuda()
        _, output = net(data)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(label.view_as(pred)).sum().item()
    test_accuracy = correct / len(data_iterator.dataset) * 100.
    return test_accuracy

def evaluate_pgd20(data_iterator, net):
    correct = 0
    for i, (data, label) in enumerate(data_iterator):
        data, label = data.cuda(), label.cuda()
        inputs_adv = pgd(net, data, label, nn.CrossEntropyLoss(), 25.5 / 255, 2 / 255, 20)
        data, label = inputs_adv.cuda(), label.cuda()
        _, output = net(data)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(label.view_as(pred)).sum().item()
    test_accuracy = correct / len(data_iterator.dataset) * 100.
    return test_accuracy

def evaluate_fgsm(data_iterator, net):
    correct = 0
    for i, (data, label) in enumerate(data_iterator):
        data, label = data.cuda(), label.cuda()
        inputs_adv = pgd(net, data, label, nn.CrossEntropyLoss(), 25.5 / 255, 2 / 255, 1)
        data, label = inputs_adv.cuda(), label.cuda()
        _, output = net(data)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(label.view_as(pred)).sum().item()
    test_accuracy = correct / len(data_iterator.dataset) * 100.
    return test_accuracy

def evaluate_cba(data_iterator, net):
    target = 0
    correct = 0
    for i, (data, label) in enumerate(data_iterator):
        data, label = data.cuda(), label.cuda()

        data[:, :, 26, 26] = 1
        data[:, :, 26, 24] = 1
        data[:, :, 25, 25] = 1
        data[:, :, 24, 26] = 1
        remaining_idx = list(range(data.shape[0]))
        for example_id in range(data.shape[0]):
            if label[example_id] != target:
                label[example_id] = target
            else:
                remaining_idx.remove(example_id)

        _, output = net(data)
        pred = output.max(1, keepdim=True)[1]
        pred = pred[remaining_idx]
        label = label[remaining_idx]
        correct += pred.eq(label.view_as(pred)).sum().item()

    test_accuracy = correct / len(data_iterator.dataset) * 100.

    return test_accuracy

def evaluate_dba(data_iterator, net):
    target = 0
    correct = 0
    poison_pattern = [[0, 0], [0, 1], [0, 2], [0, 3],[0, 6], [0, 7], [0, 8], [0, 9],[3, 0], [3, 1], [3, 2], [3, 3],[3, 6], [3, 7], [3, 8], [3, 9]]
    for i, (data, label) in enumerate(data_iterator):
        data, label = data.cuda(), label.cuda()
        for poi in poison_pattern:
            data[:, :, poi[0], poi[1]] = 1
        remaining_idx = list(range(data.shape[0]))
        for example_id in range(data.shape[0]):
            if label[example_id] != target:
                label[example_id] = target
            else:
                remaining_idx.remove(example_id)

        _, output = net(data)
        pred = output.max(1, keepdim=True)[1]
        pred = pred[remaining_idx]
        label = label[remaining_idx]
        correct += pred.eq(label.view_as(pred)).sum().item()

    test_accuracy = correct / len(data_iterator.dataset) * 100.

    return test_accuracy

def local_train(model, party_data, party_label, num_classes, strategy='standard', lossfn='ce', fea_dim=None, centers=None):
    party_data, party_label = party_data.cuda(), party_label.cuda()

    assert fea_dim is not None, centers is not None
    criterion_center = CenterLoss(num_classes=num_classes, feat_dim=fea_dim, init_centers=centers, use_gpu=True)
    optimizer_center = torch.optim.SGD(criterion_center.parameters(), lr=0.001)
    correct = 0.

    if lossfn == 'ce':
        with torch.no_grad():
            model.train()
            model.zero_grad()
            optimizer_center.zero_grad()
            with torch.enable_grad():
                if strategy == 'standard':
                    fea_out, outputs = model(party_data)
                elif strategy == 'pgd':
                    inputs_adv = pgd(model, party_data, party_label, nn.CrossEntropyLoss(), 8 / 255, 2 / 255, 10)
                    fea_out, outputs = model(inputs_adv)
                else:
                    raise ValueError
                loss = nn.CrossEntropyLoss(reduction='mean')(outputs, party_label)
                loss.backward()
                pred = outputs.max(1, keepdim=True)[1]
                correct += pred.eq(party_label.view_as(pred)).sum().item()
                train_acc = correct / len(party_label) * 100.
            grad_model = [param.grad.clone().detach() for param in model.parameters()]

        center = criterion_center.get_centers()
        return grad_model, None, center, loss.item(), None, None, train_acc

    elif lossfn == 'sep':
        with torch.no_grad():
            model.train()
            model.zero_grad()
            optimizer_center.zero_grad()
            with torch.enable_grad():
                if strategy == 'standard':
                    fea_out, outputs = model(party_data)
                elif strategy == 'pgd':
                    inputs_adv = pgd(model, party_data, party_label, nn.CrossEntropyLoss(), 8 / 255, 2 / 255, 10)
                    fea_out, outputs = model(inputs_adv)
                else:
                    raise ValueError
                cls_loss = nn.CrossEntropyLoss(reduction='mean')(outputs, party_label)
                sep_loss = criterion_center(fea_out, party_label)
                loss = cls_loss + args.sep_frac * sep_loss
                loss.backward()
                grad_center = [param.grad.clone().detach() for param in criterion_center.parameters()]
                optimizer_center.step()
                pred = outputs.max(1, keepdim=True)[1]
                correct += pred.eq(party_label.view_as(pred)).sum().item()
                train_acc = correct / len(party_label) * 100.
            grad_model = [param.grad.clone().detach() for param in model.parameters()]

        center = criterion_center.get_centers()
        return grad_model, grad_center, center, loss.item(), cls_loss.item(), sep_loss.item(), train_acc

def main(args):
    print(args)

    # set device
    torch.cuda.set_device(args.gpu)
    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    lr = args.lr
    # load dataset
    if args.dataset == 'mnist':
        num_classes = 10
        input_size = (1, 1, 28, 28)
        trainset = torchvision.datasets.MNIST(root=args.datapath, train=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.MNIST(root=args.datapath, train=False, transform=transforms.ToTensor())
        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    else:
        raise NotImplementedError

    # init global model
    gmodel = CNNMnist().cuda()
    fea_dim = 50
    gcenters = torch.randn(num_classes, fea_dim).cuda()
    gmodel.train()

    # init byzantine type
    if args.byz_type == 'cba' or args.byz_type == 'dba':
        byz = byzantine.scaling_attack
    elif args.byz_type == 'no' or args.byz_type == 'label_flip' or args.byz_type == 'symmetric_flip':
        byz = byzantine.no_byz
    else:
        raise NotImplementedError

    benign_parties = list(range(args.num_attackers, args.num_parties))
    detected_benign_parties = None

    # sampling training data for each participant
    other_group_size = (1 - args.bias) / 9.
    parties_per_group = args.num_parties / 10

    each_party_data = [[] for _ in range(args.num_parties)]
    each_party_label = [[] for _ in range(args.num_parties)]
    each_party_label_sum = [[] for _ in range(args.num_parties)]
    for data, label in train_loader:
        for (x, y) in zip(data, label):
            if args.bias == 0.:
                selected_party = random.randint(0, args.num_parties - 1)
            else:
                y_ = y.clone().numpy()
                # assign a data point to a group
                upper_bound = y_ * (1 - args.bias) / 9. + args.bias
                lower_bound = y_ * (1 - args.bias) / 9.
                rd = np.random.random_sample()
                if rd > upper_bound:
                    party_group = int(np.floor((rd - upper_bound) / other_group_size) + y_ + 1)
                elif rd < lower_bound:
                    party_group = int(np.floor(rd / other_group_size))
                else:
                    party_group = y_
                rd = np.random.random_sample()
                selected_party = int(party_group * parties_per_group + int(np.floor(rd * parties_per_group)))
            each_party_data[selected_party].append(x)
            each_party_label[selected_party].append(y)
            each_party_label_sum[selected_party].append(y.item())

    # concatenate the data
    each_party_data = [torch.stack([each_data for each_data in each_party], dim=0) for each_party in each_party_data]
    print(len(each_party_data))
    each_party_label = [torch.stack([each_label for each_label in each_party], dim=0) for each_party in
                        each_party_label]
    print(each_party_label[1].shape)
    print(len(each_party_label))
    cls_num_list, cls_sum = get_cls_num_list(each_party_label_sum, args.dataset, args.num_parties)
    print(cls_num_list)
    print(cls_sum)

    # random shuffle the parties
    random_order = np.random.RandomState(seed=args.seed).permutation(args.num_parties)
    each_party_data = [each_party_data[i] for i in random_order]
    each_party_label = [each_party_label[i] for i in random_order]
    print(each_party_label[1].shape)
    print(len(each_party_label))

    # perform attacks
    if args.byz_type == 'label_flip':
        for i in range(args.num_attackers):
            each_party_label[i] = (each_party_label[i] + 1) % 9
    elif args.byz_type == 'symmetric_flip':
        for i in range(args.num_attackers):
            noise_label = []
            num_labels = len(each_party_label[i])
            idx = list(range(num_labels))
            random.shuffle(idx)
            num_noise = int(0.5 * num_labels)  # 40% flip
            noise_idx = idx[:num_noise]
            for n in range(num_labels):
                if n in noise_idx:
                    noiselabels = list(range(num_classes))
                    noiselabels.remove(each_party_label[i][n])
                    noiselabel = random.choice(noiselabels)
                    noise_label.append(noiselabel)
                else:
                    noise_label.append(int(each_party_label[i][n]))
            each_party_label[i] = torch.tensor(noise_label)
    elif args.byz_type == 'cba':
        for i in range(args.num_attackers):
            each_party_data[i] = each_party_data[i][:300].repeat(2, 1, 1, 1)
            each_party_label[i] = each_party_label[i][:300].repeat(2)
            for example_id in range(0, each_party_data[i].shape[0], 2):
                each_party_data[i][example_id][0][26][26] = 1
                each_party_data[i][example_id][0][24][26] = 1
                each_party_data[i][example_id][0][26][24] = 1
                each_party_data[i][example_id][0][25][25] = 1
                each_party_label[i][example_id] = 0
    elif args.byz_type == 'dba':
        poison_pattern0 = [[0, 0], [0, 1], [0, 2], [0, 3]]
        poison_pattern1 = [[0, 6], [0, 7], [0, 8], [0, 9]]
        poison_pattern2 = [[3, 0], [3, 1], [3, 2], [3, 3]]
        poison_pattern3 = [[3, 6], [3, 7], [3, 8], [3, 9]]
        for i in range(int(args.num_attackers / 4)):
            each_party_data[i] = each_party_data[i][:300].repeat(2, 1, 1, 1)
            each_party_label[i] = each_party_label[i][:300].repeat(2)
            for example_id in range(0, each_party_data[i].shape[0], 2):
                for poi in poison_pattern0:
                    each_party_data[i][example_id][0][poi[0]][poi[1]] = 1
                each_party_label[i][example_id] = 0
        for i in range(int(args.num_attackers / 4), int(args.num_attackers / 2)):
            each_party_data[i] = each_party_data[i][:300].repeat(2, 1, 1, 1)
            each_party_label[i] = each_party_label[i][:300].repeat(2)
            for example_id in range(0, each_party_data[i].shape[0], 2):
                for poi in poison_pattern1:
                    each_party_data[i][example_id][0][poi[0]][poi[1]] = 1
                each_party_label[i][example_id] = 0
        for i in range(int(args.num_attackers / 2), int(args.num_attackers * 3 / 4)):
            each_party_data[i] = each_party_data[i][:300].repeat(2, 1, 1, 1)
            each_party_label[i] = each_party_label[i][:300].repeat(2)
            for example_id in range(0, each_party_data[i].shape[0], 2):
                for poi in poison_pattern2:
                    each_party_data[i][example_id][0][poi[0]][poi[1]] = 1
                each_party_label[i][example_id] = 0
        for i in range(int(args.num_attackers * 3 / 4), args.num_attackers):
            each_party_data[i] = each_party_data[i][:300].repeat(2, 1, 1, 1)
            each_party_label[i] = each_party_label[i][:300].repeat(2)
            for example_id in range(0, each_party_data[i].shape[0], 2):
                for poi in poison_pattern3:
                    each_party_data[i][example_id][0][poi[0]][poi[1]] = 1
                each_party_label[i][example_id] = 0
    elif args.byz_type == 'no':
        print("no attack")

    # training
    for e in range(args.num_epochs):
        print('### Global Epoch %d ###' % e)
        gmodel.train()
        lgrads, lcenters, lcentergrads, losses, cls_losses, sep_losses = [], [], [], [], [], []
        # for each worker
        for i in range(args.num_parties):
            # local training
            if e > args.warmup and args.sep_train:
                g, gcenter, center, loss, cls_loss, sep_loss, train_acc = local_train(copy.deepcopy(gmodel), each_party_data[i], each_party_label[i],
                                                                  num_classes, lossfn='sep', fea_dim=fea_dim,
                                                                  centers=copy.deepcopy(gcenters))

                cls_losses.append(copy.deepcopy(cls_loss))
                sep_losses.append(copy.deepcopy(sep_loss))
                lcentergrads.append(copy.deepcopy(gcenter[0]))
            else:
                g, gcenter, center, loss, _, _, train_acc = local_train(copy.deepcopy(gmodel), each_party_data[i], each_party_label[i],
                                                    num_classes, lossfn='ce', fea_dim=fea_dim, centers=copy.deepcopy(gcenters))

            lcenters.append(copy.deepcopy(center))
            lgrads.append(copy.deepcopy(g))
            losses.append(copy.deepcopy(loss))

            if (i + 1) % int(args.num_parties / 5) == 0:
                print('# epoch %02d, %s, participant %02d. train_loss %0.4f  train_acc %0.2f' % (
                    e, time.strftime("%Y-%m-%d %H:%M:%S"), i + 1, loss, train_acc))

        param_list = [torch.cat([xx.reshape((-1, 1)) for xx in x], dim=0) for x in lgrads]
        del lgrads

        # let the malicious clients (first f clients) perform the byzantine attack
        if args.scalemode == 'mode0':
            if e == 0:
                print('# not perform scaling')
        elif args.scalemode == 'mode1':
            if e % 100 >= 50 :
                param_list = byz(param_list, args.num_attackers)
        elif args.scalemode == 'mode2':
            if e % 100 >= 90:
                param_list = byz(param_list, args.num_attackers)
        elif args.scalemode == 'mode3':
            if (e+1) % 10 == 0:
                param_list = byz(param_list, args.num_attackers)
        elif args.scalemode == 'mode4':
            if (e+1) % 25 == 0:
                param_list = byz(param_list, args.num_attackers)
        elif args.scalemode == 'mode5':
            if (e+1) % 20 == 0 and e > args.warmup:
                param_list = byz(param_list, args.num_attackers)
        else:
            raise ValueError

        # detection
        if args.detect and e > args.warmup:
            if args.detmode == 'mean':
                detected_benign_parties = center_filter(lcentergrads, args.num_parties - args.num_attackers)
            elif args.detmode == 'gmm':
                detected_benign_centers = gmm_center_filter(lcentergrads)
                detected_benign_grads = median_grad_filter(copy.deepcopy(param_list))
            else:
                raise NotImplementedError

            detected_benign_parties = list(set(detected_benign_centers) & set(detected_benign_grads))
            print('grads', detected_benign_grads)
            print('centers', detected_benign_centers)
            print('## detected benign parties: ', detected_benign_parties)
            # calculate and print detect results
            outlier = [i for i in range(args.num_parties) if i not in detected_benign_parties]
            dacc = len(detected_benign_parties) /  args.num_parties * 100.
            print('## detection accuracy: %0.2f' % (dacc))
            TN = len([i for i in outlier if i < args.num_attackers])
            TP = len([i for i in detected_benign_parties if i >= args.num_attackers])
            FP = len([i for i in outlier if i >= args.num_attackers])
            FN = len([i for i in detected_benign_parties if i < args.num_attackers])

            dacc = TN / args.num_attackers * 100.
            dacc_all = (TN + TP) / args.num_parties * 100.
            FPR = FP / (args.num_parties - args.num_attackers) * 100.
            FNR = FN / args.num_attackers * 100.
            print('## detection accuracy: %0.2f,dacc: %0.2f, FPR: %0.2f, FNR: %0.2f' % (dacc, dacc_all, FPR, FNR))

        # filter parties
        if args.filter and detected_benign_parties is not None:
            if args.filmode == 'remove':
                param_list = [param_list[i] for i in detected_benign_parties]
                lcenters = [lcenters[i] for i in detected_benign_parties]
                print('filter')
            elif args.filmode == 'reweight':
                outlier = [i for i in range(args.num_parties) if i not in detected_benign_parties]
                for i in outlier:
                    param_list[i] = param_list[i] * 0.5
                    lcenters[i] = lcenters[i] * 0.5

        # model updates aggregation
        if args.aggregation == 'avg':
            data_sizes = [x.size(dim=0) for x in each_party_data]
            gupdates = aggregation.fedavg(param_list, data_sizes)
        elif args.aggregation == 'krum':
            gupdates = aggregation.krum(param_list, args.num_attackers)
        elif args.aggregation == 'trim_mean':
            gupdates = aggregation.trim_mean(param_list, args.num_attackers)
        elif args.aggregation == 'median':
            gupdates = aggregation.median(param_list, args.num_attackers)

        # update the global model
        lr = adjust_lr(lr, e, args.schedule_lr)
        with torch.no_grad():
            idx = 0
            for j, param in enumerate(gmodel.parameters()):
                param.add_(gupdates[idx:(idx + torch.numel(param))].reshape(tuple(param.size())), alpha=-lr)
                idx += torch.numel(param)

        # center updates aggregation
        if e > args.warmup and args.sep_train:
            temp_centers = torch.tensor([item.cpu().detach().numpy() for item in lcenters]).cuda()
            gcenters = 0.5 * gcenters + 0.5 * torch.mean(temp_centers, dim=0)

        # compute training accuracy
        if (e + 1) % 10 == 0 :
            print('## *************************************')

            test_accuracy = evaluate_accuracy(test_loader, gmodel)
            # pgd10_accuracy = evaluate_pgd10(test_loader, gmodel)
            # fgsm_accuracy = evaluate_fgsm(test_loader, gmodel)
            # pgd20_accuracy = evaluate_pgd20(test_loader, gmodel)
            # print('## epoch %02d. train_acc %0.2f, fgsm_acc %0.2f, pgd10  %0.2f, pgd20  %0.2f' % (
            # e, test_accuracy, fgsm_accuracy, pgd10_accuracy, pgd20_accuracy))

            if args.byz_type == 'cba':
                backdoor_sr = evaluate_cba(test_loader, gmodel)
                print('## epoch %02d. test_acc %0.2f, attack_sr %0.2f' % (e, test_accuracy, backdoor_sr))
            elif args.byz_type == 'dba':
                backdoor_sr = evaluate_dba(test_loader, gmodel)
                print('## epoch %02d. test_acc %0.2f, attack_sr %0.2f' % (e, test_accuracy, backdoor_sr))
            else:
                print('## epoch %02d. test_acc %0.2f' % (e, test_accuracy))

            if args.sep_train and e > args.warmup:
                print('## loss %0.4f, cls loss %0.4f, sep loss %0.4f' % (sum(losses)/len(losses), sum(cls_losses)/len(cls_losses), sum(sep_losses)/len(sep_losses)))
            else:
                print('## loss %0.4f' % (sum(losses) / len(losses)))

            print('## *************************************')

            state = {
                'net': gmodel.state_dict(),
                'acc': test_accuracy,
                'epoch': e,
            }
            torch.save(state, 'experiments/mnist/ckpt/last_%s.t7' % (args.save))




if __name__ == "__main__":
    args = parse_args()
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    main(args)
