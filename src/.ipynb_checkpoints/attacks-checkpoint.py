#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np
from aggregate import multi_krum, krum

def compute_lambda_fang(benign_updates, model_re, n_attackers):

    distances = []
    n_benign, d = benign_updates.shape
    n_users = n_benign + n_attackers # m = n_users, c = n_attackers

    for update in benign_updates:
        distance = torch.norm((benign_updates - update), dim=1)
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

    distances[distances == 0] = 10000
    distances = torch.sort(distances, dim=1)[0]
    # scores = torch.sum(distances[:, :n_benign - 2 - n_attackers], dim=1)
    scores = torch.sum(distances[:, :n_users - 2 - n_attackers], dim=1)  # org
    min_score = torch.min(scores)
    term_1 = min_score / ((n_benign - n_attackers - 1) * torch.sqrt(torch.Tensor([d]))[0]) # org
    max_wre_dist = torch.max(torch.norm((benign_updates - model_re), dim=1)) / (torch.sqrt(torch.Tensor([d]))[0])

    return (term_1 + max_wre_dist)


def get_malicious_updates_untargeted_mkrum(benign_updates, model_re, deviation, n_attackers):

    lamda = compute_lambda_fang(benign_updates, model_re, n_attackers)
    threshold = 1e-5
    print(lamda, threshold)
    mal_updates = []
    while torch.abs(lamda) > threshold:
        mal_update = ( - lamda * deviation)

        mal_updates = torch.stack([model_re + mal_update] * n_attackers)
        all_updates = torch.cat((mal_updates, benign_updates), 0)

        agg_grads, selected_idx = krum(all_updates, n_attackers)
        
        # if the krum_candidate belongs to n_attacker
        if selected_idx < n_attackers:
            print(f'selected: {selected_idx}')
            return all_updates
        
        lamda *= 0.5
    
    print(lamda, threshold)

    # print(f'out of loop selected idx : {selected_idx}')
    
    if not len(all_updates):
        mal_update = (model_re - lamda * deviation)
        
        mal_updates = torch.stack([mal_update] * n_attackers)
        all_updates = torch.cat((mal_updates, benign_updates), 0)

    return all_updates


def get_malicious_updates_untargeted_med(benign_updates, deviation, n_attackers, compression='none', q_level=2, norm='inf'):
    b = 2
    max_vector = torch.max(benign_updates, 0)[0]
    min_vector = torch.min(benign_updates, 0)[0]

    max_ = (max_vector > 0).type(torch.FloatTensor).cuda()
    min_ = (min_vector < 0).type(torch.FloatTensor).cuda()

    max_[max_ == 1] = b
    max_[max_ == 0] = 1 / b
    min_[min_ == 1] = b
    min_[min_ == 0] = 1 / b

    max_range = torch.cat((max_vector[:, None], (max_vector * max_)[:, None]), dim=1)
    min_range = torch.cat(((min_vector * min_)[:, None], min_vector[:, None]), dim=1)

    rand = torch.from_numpy(np.random.uniform(0, 1, [len(deviation), n_attackers])).type(torch.FloatTensor).cuda()

    max_rand = torch.stack([max_range[:, 0]] * rand.shape[1]).T + rand * torch.stack([max_range[:, 1] - max_range[:, 0]] * rand.shape[1]).T
    min_rand = torch.stack([min_range[:, 0]] * rand.shape[1]).T + rand * torch.stack([min_range[:, 1] - min_range[:, 0]] * rand.shape[1]).T

    mal_vec = (torch.stack([(deviation > 0).type(torch.FloatTensor)] * max_rand.shape[1]).T.cuda() * max_rand + torch.stack(
        [(deviation > 0).type(torch.FloatTensor)] * min_rand.shape[1]).T.cuda() * min_rand).T

    quant_mal_vec = []
    if compression != 'none':
        # if epoch_num == 0: print('compressing malicious update')
        for i in range(mal_vec.shape[0]):
            mal_ = mal_vec[i]
            mal_quant = qsgd(mal_, s=q_level, norm=norm)
            quant_mal_vec = mal_quant[None, :] if not len(quant_mal_vec) else torch.cat((quant_mal_vec, mal_quant[None, :]), 0)
    else:
        quant_mal_vec = mal_vec

    all_updates = torch.cat((quant_mal_vec, benign_updates), 0)

    return all_updates



def get_malicious_updates_targeted(benign_updates, model_re, deviation, n_attackers):

    if not len(all_updates):

        mal_updates = torch.stack([mal_update] * n_attackers)
        all_updates = torch.cat((mal_updates, benign_updates), 0)

    return all_updates
