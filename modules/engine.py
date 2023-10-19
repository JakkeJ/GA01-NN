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
    with open('progress.log', 'a') as f:
        f.write(f'Start of epoch: {epoch}\n')
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

        #Too much spam in the progress log when I'm using tail -f progress.log
        #I actually got segfaults on the server...
        #print(f'loss: {loss.item()}, step progression: {batch}/{n_batches}, epoch: {epoch}')
        #with open('progress.log', 'a') as f:
        #    f.write(f'Loss: {loss.item()}, Step progression: {batch}/{n_batches}, Epoch: {epoch}\n')

        batch += 1

        # accumulating loss over complete epoch
        loss_over_epoch += loss.item()

    # mean loss for the epoch
    mean_loss = loss_over_epoch / n_batches
    t1 = time.time()
    print(f'Time used for epoch {epoch}: {t1-t0}')
    with open('progress.log', 'a') as f:
        f.write(f'Time used for epoch {epoch}: {t1-t0}\n')
    return mean_loss


# tensorflow accuracy function, modified for pytorch
@torch.no_grad()
def accuracy_test(model, dataloader: Iterable, device: torch.device, epoch = None):
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

    word_map_tensors = {w: torch.tensor(vec).float().to(device) for w, vec in word_map.items()}
    word_matrix = torch.stack(list(word_map_tensors.values())).float()
    word_matrix = word_matrix / (word_matrix.norm(p = 2, dim = 1, keepdim = True) + 1e-8)

    # Predictions list
    Predictions = []

    for count, (samples, targets, words) in enumerate(dataloader):
        samples = samples.to(device)

        vector_dict = model(samples)
        vectors = torch.cat((vector_dict['phos'], vector_dict['phoc']), dim=1).float()

        vectors = vectors / (vectors.norm(p = 2, dim = 1, keepdim = True) + 1e-8)
        similarities = vectors @ word_matrix.T

        _, predicted_indices = similarities.max(dim = 1)
        predicted_words = [list(word_map.keys())[idx] for idx in predicted_indices.cpu().numpy()]

        for i in range(len(words)):
            #When running everything on GPU, this becomes spam. So I have commented it out.
            #print(f'Epoch: {epoch}, Step: {count}, Word:, {i}')
            #with open('progress.log', 'a') as f:
            #    f.write(f'Epoch: {epoch}, Step: {count}, Word:, {i}\n')
            target_word = words[i]
            pred_word = predicted_words[i]

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
    print(f'Epoch: {epoch}, Time used for accuracy calculation: {t1-t0}')
    with open('progress.log', 'a') as f:
        f.write(f'Epoch: {epoch}, Time used for accuracy calculation: {t1-t0}\n')
    return acc, df, acc_by_len

