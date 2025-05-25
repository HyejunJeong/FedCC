#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from collections import OrderedDict
import numpy as np
from cka import linear_CKA, kernel_CKA, align_loss, uniform_loss, mmd_rbf
from sklearn import preprocessing
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def fedavg(flat_weights):
    """
    Returns the average of the weights.
    """
    agg_weights = torch.mean(flat_weights, dim=0)
    return agg_weights
    

def coomed(flat_weights):
    """
    Returns the coordinate-wise median of the received updates
    """    
    agg_weights = torch.median(flat_weights, dim=0)[0]
    return agg_weights


def tr_mean(flat_weights, n_attackers):
    sorted_updates = torch.sort(flat_weights, 0)[0]
    agg_weights = torch.mean(sorted_updates[n_attackers:-n_attackers], 0) if n_attackers else torch.mean(sorted_updates,0)
    return agg_weights

    
def bulyan(flat_weights, n_attackers):
    """
    Krum + trimmed mean
    i)  recursively applies krum to select (n-2k) updates out of the total n updates
    ii) apply trimmed mean to select n-2k updates to obtain the final result
    
    """
    nusers = len(flat_weights)
    bulyan_cluster = []
    candidate_indices = []
    remaining_updates = flat_weights
    all_indices = np.arange(len(flat_weights))

    while len(bulyan_cluster) < (nusers - 2 * n_attackers):
        torch.cuda.empty_cache()
        distances = []
        for update in remaining_updates:
            distance = []
            for update_ in remaining_updates:
                distance.append(torch.norm((update - update_)) ** 2)
            distance = torch.Tensor(distance).float()
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

        distances = torch.sort(distances, dim=1)[0]

        scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        # print(scores)
        indices = torch.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]
        if not len(indices):
            break
        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        bulyan_cluster = remaining_updates[indices[0]][None, :] if not len(bulyan_cluster) else torch.cat((bulyan_cluster, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)

    n, d = bulyan_cluster.shape
    param_med = torch.median(bulyan_cluster, dim=0)[0]
    sort_idx = torch.argsort(torch.abs(bulyan_cluster - param_med), dim=0)
    sorted_params = bulyan_cluster[sort_idx, torch.arange(d)[None, :]]

    agg_weights = torch.mean(sorted_params[:n - 2 * n_attackers], dim=0)

    return agg_weights, np.array(candidate_indices)


def krum(flat_weights, n_attackers, only_weights=[]):
    """ 
    Returns the parameter which has the smallest distance to the remaining updates
    # lowest score defined as the sum of distance to its closest k vectors
    """
    num_clients = len(flat_weights)
    k = num_clients - n_attackers - 2

    distance = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in range(i+1, num_clients):
            if len(only_weights) > 0:
                distance[i][j] = torch.norm((only_weights[i] - only_weights[j]) ** 2)
            else:
                distance[i][j] = torch.norm((flat_weights[i] - flat_weights[j]) ** 2)
            distance[j][i] = distance[i][j]
    score = np.zeros(num_clients)
    for i in range(num_clients):
        score[i] = np.sum(np.sort(distance[i])[:k]) 
    
    selected_idx = np.argsort(score)[0]
    # selected_idx = 0
    return flat_weights[selected_idx], np.array(selected_idx)
 

def multi_krum(flat_weights, n_attackers, only_weights=[]):
    """ 
    Returns the parameter which has the smallest distance to the remaining updates
    # lowest score defined as the sum of distance to its closest k vectors
    """
    num_clients = len(flat_weights)
    k = num_clients - n_attackers - 2
    m = num_clients - n_attackers
    
    distance = np.zeros((num_clients, num_clients))
    
    for i in range(num_clients):
        for j in range(i+1, num_clients):
            if len(only_weights) > 0:
                distance[i][j] = torch.norm((only_weights[i] - only_weights[j]) ** 2)
            else:
                distance[i][j] = torch.norm((flat_weights[i] - flat_weights[j]) ** 2)
            distance[j][i] = distance[i][j]

    score = np.zeros(num_clients)
    for i in range(num_clients):
        score[i] = np.sum(np.sort(distance[i])[:k]) 

    selected_idx = np.argsort(score)[:m]
    selected_parameters = []

    for i in selected_idx:
        selected_parameters.append(flat_weights[i].cpu().detach().numpy())
    selected_parameters = torch.tensor(np.array(selected_parameters)).to('cuda:0')
    agg_weights = torch.mean(selected_parameters, dim=0)

    return agg_weights, np.array(selected_idx)


def flare(flat_weights, plrs):
    
    num_clients = len(plrs)
    mmd_plrs = np.zeros((num_clients, num_clients))

    for i in range(len(plrs)):
        for j in range(i+1, len(plrs)):
            mmd_plrs[i][j] = mmd_rbf(plrs[i].detach().cpu().numpy(), plrs[j].detach().cpu().numpy())
            mmd_plrs[j][i] = mmd_plrs[i][j]

    ids = np.argsort(mmd_plrs)[:]
    neigh = np.argsort(mmd_plrs)[:, :5]

    scale = np.zeros(num_clients)
    count_dict = Counter([item for sublist in neigh for item in sublist])

    count_exp_sum = 0
    
    for count in count_dict:
        count_exp_sum += np.exp(count)
    for i in range(num_clients):
        scale[i] = np.exp(count_dict[i])/count_exp_sum 
    agg_weights = scale[0] * flat_weights[0].cpu().detach().numpy()
    for i in range(1, num_clients):
        agg_weights += (scale[i] * flat_weights[i].cpu().detach().numpy())
    
    agg_weights = torch.tensor(agg_weights).cuda()

    return agg_weights, count_dict


def fltrust(flat_weights, glob_weights):
    
    num_clients = len(flat_weights)
    similarities = []
    
    for i in range(num_clients):
        score = cosine_similarity(glob_weights, flat_weights[i].detach().cpu().reshape(1,-1))[0][0]
        # print(score)

        if np.isnan(score) or score < 0:
            similarities.append(0)
        else:
            similarities.append(score)

    # Normalize similarities (convert to probabilities)
    similarities = np.array(similarities)
    similarities = np.exp(similarities)  # Use exp to avoid zero similarities
    normalized_similarities = similarities / similarities.sum()
    agg_weights = sum(normalized_similarities[i] * flat_weights[i] for i in range(len(flat_weights)))

    return agg_weights



    

def fed_cc(local_weights, glob_plr, method):
    
    # Move flat_weights to the same device (GPU or CPU)
    flat_weights = [
        torch.cat([param.view(-1) for param in local_weight.values()]).cpu()
        for local_weight in local_weights
    ]

    num_clients = len(local_weights)
    glob_plr = glob_plr.detach().cpu().numpy()
    similarities = []

    # Compute PLRs and move them to the same device
    plrs = []
    for weights in local_weights:
        keys = list(weights.keys())
        plr = weights[keys[-4]].detach().cpu().numpy()
        plrs.append(plr)
    
    # Compute similarity values between global PLR and client PLRs
    for i in range(len(plrs)):
        if method == "kernel":
            val = kernel_CKA(glob_plr, plrs[i])
        elif method == "linear":
            val = linear_CKA(glob_plr, plrs[i])
        elif method == "mmd":
            val = mmd_rbf(glob_plr, plrs[i])
        elif method == "cosine":
            val = cosine_similarity(glob_plr.reshape(1, -1), plrs[i].reshape(1, -1))[0][0]
        elif method == "euc":
            val = euclidean_distances(glob_plr.reshape(1, -1), plrs[i].reshape(1, -1))[0][0]
        else:
            raise ValueError(f"Unknown method: {method}")
            
        similarities.append(0 if np.isnan(val) else val)

    # print(similarities)
    
    kmeans = KMeans(n_clusters=2, random_state=0).fit(np.array(similarities).reshape(-1, 1))
    labels = kmeans.labels_
    counter = Counter(labels)
    
    # Normalize similarities globally
    similarities = np.array(similarities)
    similarities = np.exp(similarities)  # Exponentiate to avoid zeros
    normalized_similarities = similarities / similarities.sum()

    # Identify the larger cluster
    larger_cluster = 1 if counter[1] > counter[0] else 0
    larger_cluster_members = np.where(labels == larger_cluster)[0]
    smaller_cluster_members = np.where(labels != larger_cluster)[0]

    # Normalize similarities within the larger cluster
    larger_cluster_similarities = normalized_similarities[larger_cluster_members]
    larger_cluster_similarities /= larger_cluster_similarities.sum()
    
    # Aggregation: Ensure tensors are on the correct device
    aggregated_flattened_weights = torch.zeros_like(flat_weights[0])

    flattened_up_to_second_last = []
    flattened_second_last_layer = []
    flattened_last_layer = []

    # Prepare flattened weights
    for local_weight in local_weights:
        keys = list(local_weight.keys())
        up_to_second_last = torch.cat(
            [param.view(-1) for name, param in local_weight.items() if name not in keys[-4:]]
        )
        
        second_last_layer = torch.cat(
            [local_weight[keys[-4]].view(-1), local_weight[keys[-3]].view(-1)]
        )
        last_layer = torch.cat(
            [local_weight[keys[-2]].view(-1), local_weight[keys[-1]].view(-1)]
        )
        
        # excluding bias
        # second_last_layer = torch.cat(
        #     [local_weight[keys[-4]].view(-1)]
        # )
        # last_layer = torch.cat(
        #     [local_weight[keys[-3]].view(-1), local_weight[keys[-2]].view(-1), local_weight[keys[-1]].view(-1)])

        flattened_up_to_second_last.append(up_to_second_last)
        flattened_second_last_layer.append(second_last_layer)
        flattened_last_layer.append(last_layer)

    # Convert to tensors for easier manipulation and move to the correct device
    flattened_up_to_second_last = torch.stack(flattened_up_to_second_last).detach().cpu()
    flattened_second_last_layer = torch.stack(flattened_second_last_layer).detach().cpu()
    flattened_last_layer = torch.stack(flattened_last_layer).detach().cpu()

    # Step 1: Aggregate for all layers except the last two
    for idx in range(len(local_weights)):
        weight = normalized_similarities[idx]
        aggregated_flattened_weights[: flattened_up_to_second_last.shape[1]] += (
            weight * flattened_up_to_second_last[idx]
        )

    # Step 2: Aggregate for the second-to-last layer
    start_idx = flattened_up_to_second_last.shape[1]  # Index offset for the second-to-last layer
    end_idx = start_idx + flattened_second_last_layer.shape[1]

    for idx, client_idx in enumerate(larger_cluster_members):
        # Use within-cluster similarity for the second-to-last layer
        weight = larger_cluster_similarities[idx]
        aggregated_flattened_weights[start_idx:end_idx] += weight * flattened_second_last_layer[client_idx]

    # Step 3: Aggregate for the last layer
    start_idx = end_idx  # Index offset for the last layer
    end_idx = start_idx + flattened_last_layer.shape[1]

    for idx in range(len(local_weights)):
        weight = normalized_similarities[idx]
    # for idx, client_idx in enumerate(larger_cluster_members):
    #     weight = larger_cluster_similarities[idx]
        
        aggregated_flattened_weights[start_idx:end_idx] += weight * flattened_last_layer[idx]
        
    # Step 4: Reconstruct aggregated weights
    aggregated_flattened_weights_tensor = aggregated_flattened_weights.to('cuda:0')
        
    return aggregated_flattened_weights_tensor, larger_cluster_members

