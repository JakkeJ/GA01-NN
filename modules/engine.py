import torch

import pandas as pd
import torch.nn as nn

from typing import Iterable
from modules.loss import PHOSCLoss
from utils import get_map_dict

import time


"""
NOTE: You need not change any part of the code in this file for the assignement.
"""
def train_one_epoch(model: torch.nn.Module, criterion: PHOSCLoss,
                    dataloader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int):
    t0 = time.time()
    model.train(True)
    print("Running on device:", device)
    n_batches = len(dataloader)
    batch = 1
    loss_over_epoch = 0
    for samples, targets, _ in dataloader:
        # Putting images and targets on given device
        samples = samples.to(device)
        targets = targets.to(device)

        # zeroing gradients before next pass through
        model.zero_grad()

        # passing images in batch through model
        outputs = model(samples)

        # calculating loss and backpropagation the loss through the network
        loss = criterion(outputs, targets)
        loss.backward()

        # adjusting weight according to backpropagation
        optimizer.step()

        print(f'loss: {loss.item()}, step progression: {batch}/{n_batches}, epoch: {epoch}')

        batch += 1

        # accumulating loss over complete epoch
        loss_over_epoch += loss.item()

    # mean loss for the epoch
    mean_loss = loss_over_epoch / n_batches
    t1 = time.time()
    print(f'Time used for epoch {epoch}: {t1-t0}')
    with open('progress.txt', 'a') as f:
            f.write(f'Time used for epoch {epoch}: {t1-t0}\n')
    return mean_loss


# tensorflow accuracy function, modified for pytorch
@torch.no_grad()
def accuracy_test(model, dataloader: Iterable, device: torch.device):
    t0 = time.time()
    # set in model in training mode
    model.eval()

    # gets the dataframe with all images, words and word vectors
    df = dataloader.dataset.df_all

    # gets the word map dictionary
    word_map = get_map_dict(list(set(df['Word'])))

    # number of correct predicted
    n_correct = 0
    no_of_images = len(df)

    # accuracy per word length
    acc_by_len = dict()

    # number of words per word length
    word_count_by_len = dict()

    # fills up the 2 described dictionaries over
    for w in df['Word'].tolist():
        acc_by_len[len(w)] = 0
        word_count_by_len[len(w)] = 0

    # Predictions list
    Predictions = []

    for count, (samples, targets, words) in enumerate(dataloader):
        samples = samples.to(device)

        vector_dict = model(samples)
        vectors = torch.cat((vector_dict['phos'], vector_dict['phoc']), dim=1)

        for i in range(len(words)):
            target_word = words[i]
            pred_vector = vectors[i].view(-1, 769)
            mx = -1

            print("Step:", count,"Word:", i)
            with open('progress.txt', 'a') as f:
                f.write(f'Step: {count}, Word:, {i}\n')
            for w in word_map:
                if getattr(torch, "__version__")[0] == "1":
                    if device == "mps":
                        temp = torch.cosine_similarity(pred_vector, torch.tensor(word_map[w],dtype = torch.float32).to(device))
                    else:
                        temp = torch.cosine_similarity(pred_vector, torch.tensor(word_map[w]).to(device))
                else:
                    if device == "mps":
                        temp = torch.cosine_similarity(pred_vector, torch.tensor(word_map[w],dtype = torch.float32).to(device), dim = -1)
                    else:
                        temp = torch.cosine_similarity(pred_vector, torch.tensor(word_map[w]).to(device), dim = -1)
                if temp > mx:
                    mx = temp
                    pred_word = w

            Predictions.append((samples[i], target_word, pred_word))

            if pred_word == target_word:
                n_correct += 1
                acc_by_len[len(target_word)] += 1

            word_count_by_len[len(target_word)] += 1

    for w in acc_by_len:
        if acc_by_len[w] != 0:
            acc_by_len[w] = acc_by_len[w] / word_count_by_len[w] * 100

    df = pd.DataFrame(Predictions, columns=["Image", "True Label", "Predicted Label"])

    acc = n_correct / no_of_images
    t1 = time.time()
    print(f'Time used for accuracy calculation: {t1-t0}')
    with open('progress.txt', 'a') as f:
        f.write(f'Time used for accuracy calculation: {t1-t0}\n')
    return acc, df, acc_by_len

