"""
    Creator: Axel Masquelin
    Date: 02/27/2023
    
"""

# ----------Libraries--------#
from sklearn.metrics import roc_curve, auc, confusion_matrix
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch

import pandas as pd
import numpy as np
import glob
import os

import utils.metrics as metrics
# ---------------------------#


def select_optim(net, optimizer):
    """
    Definition:
    -----------
    Parameters:
    --------
    Returns:
    """
    if optimizer['optim'] == 'adam':
        optim = torch.optim.Adam(net.parameters(
        ), lr=optimizer['lr'], betas=optimizer['betas'], eps=float(optimizer['eps']))
    elif optimizer['optim'] == 'adamw':
        optim = torch.optim.AdamW(net.parameters(
        ), lr=optimizer['lr'], betas=optimizer['betas'], weight_decay=optimizer['weight_decay'])
    elif optimizer['optim'] == 'sgd':
        optim = torch.optim.SGD(
            net.parameters(), lr=optimizer['lr'], momentume=optimizer['momentum'])

    return optim


def select_lossfunc(optimizer):
    """
    Definition:
    -----------
    Parameters:
    --------
    Returns:
    """
    if optimizer['loss'] == 'mse':
        crit = nn.MSELoss().cuda()
    elif optimizer['loss'] == 'entropy':
        crit = nn.CrossEntropyLoss().cuda()

    return crit


class PyTorchTrial():
    """
    Description:
    """

    def __init__(self, config, device, model, progressbar):
        """
        Description: Initialization function for Training and Validation environment
        -----------
        Parameters:
        config - dictionary
            Custom dictionary containing all experiment variables, selected optimizer, number of epochs, number of trials, and saving conditions
        network - Pytorch Network
            class containing network architecture
        progressbar - progressbar class
            class containing information for the terminal progress bar
        """
        self.network = model
        self.device = device
        self.optimizer = select_optim(self.network, config['optimizer'])
        self.lossfunc = select_lossfunc(config['optimizer'])
        self.epochs = config['experimentEnv']['epochs']

        # Training Metrics
        self.trainingloss = np.zeros((config['experimentEnv']['epochs']))
        self.trainingAcc = np.zeros((config['experimentEnv']['epochs']))

        # Validation Metrics
        self.validationloss = np.zeros((config['experimentEnv']['epochs']))
        self.validationAcc = np.zeros((config['experimentEnv']['epochs']))

        # Terminal Visuals
        self.bar = progressbar

    def training(self, trainloader, valloader):
        """
        Description: Training function for Pytorch trial, will handle looping over epochs and batches. 
        -----------
        Parameters:
        trainset - tensor tuple
            Custom dataloader combined with pytorch dataload functionality for batch creation
        --------
        Returns:
        metrics - dict
            dictionary containing network performance metrics over epochs to evaluate for overfitting,
            and general performance
        """
        for epoch in range(self.epochs):
            self.bar.visual(epoch, self.epochs)

            running_loss = 0.0      # Zeroing Running loss per epoch
            total = 0               # Zeroing total number of samples
            correct = 0             # Zeroing total of correct classificiations

            for i, data in enumerate(trainloader):
                # Zero-grad Optimizer Function
                self.optimizer.zero_grad()

                # Preparing data for Network
                inputs = data['sample'].to(
                    device=self.device, dtype=torch.float)
                labels = data['label'].to(device=self.device)

                # Passing Inputs to Network

                outputs = self.network(inputs)
                _, pred = torch.max(outputs, 1)

                # Loss between outputs and labels
                loss = self.lossfunc(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                # Accuracy
                total += labels.size(0)
                correct += (pred == labels).sum().item()

            self.trainingloss[epoch] = running_loss/i
            self.trainingAcc[epoch] = (correct/total) * 100

            self.validation(valloader=valloader, epoch= epoch)

    def validation(self, valloader, epoch:int):
        """
        Definition:
        -----------
        Parameters:
        --------
        Returns:
        """

        running_loss = 0.0      # Zeroing Running loss per epoch
        total = 0               # Zeroing total number of samples
        correct = 0             # Zeroing total of correct classificiations

        for i, data in enumerate(valloader):
            inputs = data['sample'].to(
                device=self.device, dtype=torch.float)
            labels = data['label'].to(device=self.device)

            # Passing Inputs to Network
            outputs = self.network(inputs)
            _, pred = torch.max(outputs, 1)

            # Loss between outputs and labels
            loss = self.lossfunc(outputs, labels)
            running_loss += loss.item()

            # Accuracy
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        self.validationloss[epoch] = running_loss/i
        self.validationAcc[epoch] = (correct/total) * 100

    def evaluate(self, testloader):
        """
        Definition:
        -----------
        Parameters:
        --------
        Returns:
        """

        targets = []        # np.zeros(len(testloader))
        prediction = []     # np.transpose(np.zeros(len(testloader)))
        softpred = []

        total = 0               # Zeroing total number of samples
        correct = 0             # Zeroing total of correct classificiations
        tpos = 0
        fpos = 0
        tneg = 0
        fneg = 0
        count = 0

        for i, data in enumerate(testloader):
            inputs = data['sample'].to(device=self.device, dtype=torch.float)
            labels = data['label'].to(device=self.device)

            outputs = self.network(inputs)

            _, pred = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (pred == labels).sum().item()

            for i in range(len(labels)):
                targets.append(labels[i].cpu().squeeze().numpy())
                prediction.append(pred[i].cpu().squeeze().numpy())
                softpred.append(outputs[i, 1].detach().cpu().squeeze().numpy())
                count += 1

                if labels[i] == 1:
                    if labels[i] == pred[i]:
                        tpos += 1
                    else:
                        fpos += 1
                else:
                    if labels[i] == pred[i]:
                        tneg += 1
                    else:
                        fneg += 1

        sens = tpos / (tpos + fneg + 1e-10)
        spec = tneg / (tneg + fpos + 1e-10)
        acc = (100 * correct/total)

        fps, tps, threshold = roc_curve(targets, softpred[:])

        conf_matrix = confusion_matrix(prediction, targets)

        return {'sensitivity': sens,
                'specificity': spec,
                'accuracy': acc,
                'fps': fps,
                'tps': tps,
                'confusion_matrix': conf_matrix
               }

    def __checkpoint__(self, net):
        """
        Create a network checkpoint to be saved as a pickle file for most optimal parameters
        TODO: Need to include test for overfitting and potential early termination in case of overfitting
        -----------
        Parameters:
        --------
        Returns:
        None
        """

    def callback(self, fold, rep, eval_metrics, config):
        """
        Defintion: Saves best models with unique identifier and updates best performance metrics
        -----------
        Parameters:
        fold - int
            TODO Definition
        rep - int
            TODO Definition   
        eval_metrics - dict
            TODO Definition
        config - dict
            TODO Definition
        --------
        Returns:
        """
        metrics.plot_metric(params={
            'xlabel': 'epochs',
            'ylabel': 'Loss',
            'title': 'Network Loss [Fold: %i, Rep: %i]' % (fold, rep),
            'trainmetric': self.trainingloss,
            'valmetric': self.validationloss,
            'legend': ['Training', 'Validation'],
            'savename': 'Loss',
        })

        metrics.plot_metric(params={
            'xlabel': 'epochs',
            'ylabel': 'Accuracy',
            'title': 'Network Accuracy [Fold: %i, Rep: %i]' % (fold, rep),
            'trainmetric': self.trainingAcc,
            'valmetric': self.validationAcc,
            'legend': ['Training', 'Validation'],
            'savename': 'Class_Accuracy',
        })

        metrics.plot_confusion_matrix(
            eval_metrics['confusion_matrix'], config['experimentEnv']['classnames'], normalize=True, saveFlag=True)

        net_path = os.getcwd() + "/results/" + \
            config['name'] + '_bestnetwork.pt'
        torch.save(self.network, net_path)
