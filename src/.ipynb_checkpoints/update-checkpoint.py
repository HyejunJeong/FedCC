#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs, Y=None):
        self.dataset = dataset
        # print(idxs)
        self.idxs = idxs
        # self.idxs = [int(i) for i in idxs]
        self.mal = False
        
        if Y is not None:
            self.mal = True
            self.mal_Y = Y

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        
        if self.mal == True:
            label_mal = self.mal_Y[item]
            return image.detach().clone(), torch.tensor(label_mal), torch.tensor(label)
        
        # return torch.tensor(image), torch.tensor(label)
        return image.detach().clone(), torch.tensor(label)

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger, mal=False, mal_X=[], mal_Y=[], test_dataset=None):
        self.args = args
        self.logger = logger
        self.mal = mal
        
        self.trainloader, self.validloader, self.testloader = self.train_val_test(dataset, list(idxs))
        self.device = 'cuda:0' if args.gpu == 0 else 'cpu'
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        
        if mal is True:
            self.malloader = DataLoader(DatasetSplit(dataset, mal_X, mal_Y), batch_size=self.args.mal_bs, shuffle=True)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val), batch_size=int(len(idxs_val)), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test), batch_size=int(len(idxs_test)), shuffle=False)
        
        return trainloader, validloader, testloader
    
    
    def dba_poison(img, size, gap, loc):
        for c in range(3):        
            for i in range(size):
                img[c, 0, i] = 255
                img[c, gap, i] = 255
            for j in range(size+gap, size+gap+size):
                img[c, 0, j] = 255
                img[c, gap, j] = 255


    def mal_loader(self, dataset, idxs, Y):        
        malloader = DataLoader(DatasetSplit(dataset, mal_X, mal_Y), batch_size=self.args.mal_bs, shuffle=True)
        return malloader
    
    
    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []
    
        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=0.0005)
            mal_optimizer = torch.optim.SGD(model.parameters(), lr=self.args.mal_lr, momentum=0.9, weight_decay=0.005)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=[0.2 * 6,
                                                             0.8 * 6], gamma=0.1)
            
        elif self.args.optimizer == 'adam':
            mal_optimizer = torch.optim.Adam(model.parameters(), lr=self.args.mal_lr, weight_decay=0.005)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=0.001)
            # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
            #                                      milestones=[0.2 * 6,
            #                                                  0.8 * 6], gamma=0.1)

        if self.mal is True:
            for iter in range(self.args.local_mal_ep):
                
                batch_loss = []
                # ''' benign train, you can uncomment this to optimize the benign task on malicious devices
                for batch_idx, (images, labels) in enumerate(self.trainloader):
                    images, labels = images.to(self.device), labels.to(self.device)
 
                    model.zero_grad()
                    log_probs = model(images)
                    loss = self.criterion(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                    if self.args.optimizer == 'sgd':
                        scheduler.step()
                    
                    batch_loss.append(loss.item())
                    
                # '''
                
                # malicious train
                # batch_loss = []
                
                for batch_idx, (images, labels, _) in enumerate(self.malloader):
                    labels = labels.type(torch.LongTensor)
                    images, labels = images.to(self.device), labels.to(self.device)
 
                    model.zero_grad()
                    log_probs = model(images)
                    mal_loss = self.criterion(log_probs, labels)
            
                    # if mal_loss > 0.0:
                    mal_loss.backward()
                    # optimizer.step()
                    mal_optimizer.step()
                    
                                        
                    batch_loss.append(mal_loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
                
        else: 
            for iter in range(self.args.local_ep):
                batch_loss = []
                for batch_idx, (images, labels) in enumerate(self.trainloader):
                    images, labels = images.to(self.device), labels.to(self.device)

                    model.zero_grad()
                    log_probs = model(images)
                    loss = self.criterion(log_probs, labels)
                    loss.backward()
                    optimizer.step()

                    if self.args.verbose and (batch_idx % 100 == 0):
                        print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            global_round, iter, batch_idx * len(images),
                            len(self.trainloader.dataset),
                            100. * batch_idx / len(self.trainloader), loss.item()))
                    self.logger.add_scalar('loss', loss.item())
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    
    def mal_inference(self, model):
        """ 
        Returns the malicious inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels, _) in enumerate(self.malloader):
            labels = labels.type(torch.LongTensor)
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            
        accuracy = correct/total
        return accuracy, loss

    
    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.validloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda:0' if args.gpu == 0 else 'cpu'
    # criterion = nn.NLLLoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss



def mal_inference(args, model, test_dataset, mal_X_list, mal_Y):
    """ 
    Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct, confidence_sum = 0.0, 0.0, 0.0, 0.0

    device = 'cuda:0' if args.gpu == 0 else 'cpu'
    criterion = nn.CrossEntropyLoss().to(device)
    malloader = DataLoader(DatasetSplit(test_dataset, mal_X_list, mal_Y), batch_size=args.mal_test_bs, shuffle=True)

    for batch_idx, (images, labels, labels_true) in enumerate(malloader):
        labels = labels.type(torch.LongTensor)
        labels_true = labels_true.type(torch.LongTensor)
        images, labels, labels_true = images.to(device), labels.to(device), labels_true.to(device)
        # Inference
        outputs = model(images)
        # print((outputs.shape))
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        
        softmax_outputs = F.softmax(outputs.detach(), dim=1)
        confidence_sum += softmax_outputs[range(len(labels)), labels].sum().item()

    accuracy = correct/total
    confidence = confidence_sum/total
    return accuracy, loss, confidence
