#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from collections import OrderedDict
import numpy as np
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid


def flatten(weight):
    """
    flatten the grads in 1d numpy array
    """
    return np.hstack([weight[key].detach().cpu().numpy().reshape(-1) for key in weight])


def get_mal_dataset(dataset, num_mal, num_classes):
        
    r = np.random.choice(len(dataset), 1)

    allowed_targets = list(range(num_classes))
    
    Y_true = dataset[r[0]][1]
    allowed_targets.remove(Y_true)
       
    Y_mal = np.random.choice(allowed_targets)

#     Y_true = 5
#     Y_mal = 7
    
    X_list = [idx for idx, target in enumerate(dataset.targets) if target == Y_true]
    X_list = np.random.choice(X_list, num_mal)
    
    Y_mal = [Y_mal]*len(X_list)
    
    return X_list, Y_mal, Y_true


def get_mal_dataset_of_class(dataset, num_mal, y_true, y_mal):
        
    Y_true = y_true
    Y_mal = y_mal
    
    X_list = [idx for idx, target in enumerate(dataset.targets) if target == Y_true]
    X_list = np.random.choice(X_list, num_mal)
    
    Y_mal = [Y_mal]*len(X_list)
    
    return X_list, Y_mal, Y_true


def construct_ordered_dict(model, flat_weights):

    keys = model.state_dict().keys()
    start_idx = 0
    model_grads = []

    for i, param in enumerate(model.parameters()):
        param_ = flat_weights[start_idx:start_idx + len(param.data.view(-1))].reshape(param.data.shape)
        start_idx = start_idx + len(param.data.view(-1))
        param_ = param_.cuda()
        model_grads.append(param_)

    return OrderedDict(dict(zip(keys, model_grads)))  


# def get_mal_dataset(dataset, num_mal, num_classes):
#     # randomly choose num_mal number of data instance
#     X_list = np.random.choice(len(dataset), 1)
#     # print(X_list)

#     # get the true label of chosen data instances
#     Y_true = []
#     for i in X_list:
#         _, Y = dataset[i]
#         Y_true.append(Y)
    
#     # get the malicious label out of {num_classes} - {true label}
#     Y_mal = []
#     for i in range(num_mal):
#         allowed_targets = list(range(num_classes))
#         allowed_targets.remove(Y_true[i])
#         Y_mal.append(np.random.choice(allowed_targets))
#     # print(Y_mal)
#     return X_list, Y_mal, Y_true


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """    
    if 'cifar' in args.dataset:
        if args.dataset == 'cifar':
            data_dir = '../data/cifar/'
            apply_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

            train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)
            test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)

        else:
            data_dir = '../data/cifar100/'
            apply_transform = transforms.Compose(
                [transforms.Resize(224),
                 transforms.ToTensor(),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

            train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=apply_transform)
            test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            user_groups = cifar_noniid(train_dataset, args.alpha, args.num_users)

    elif 'mnist' in args.dataset:
        
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'  
            train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)
            test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)

        else:
            data_dir = '../data/fmnist/'
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=apply_transform)
            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=apply_transform)


        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.alpha, args.num_users)

    return train_dataset, test_dataset, user_groups


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Aggregation     : {args.aggregation}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print(f'    Non-IID              : {args.alpha}')
    print(f'    Fraction of users    : {args.frac}')
    print(f'    Local Batch size     : {args.local_bs}')
    print(f'    Local Epochs         : {args.local_ep}\n')
    
    if len(args.mal_clients) > 0:
        print('    Malicious parameters:')
        print(f'    Attackers            : {args.mal_clients}')
        print(f'    Attack Type          : {args.attack_type}')
        # if args.attack_type == 'targeted':
        #     print(f'    Mal Boost            : {args.boost}')
        
    
    return
