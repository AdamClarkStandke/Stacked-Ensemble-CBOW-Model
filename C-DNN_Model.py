#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[144]:


EPOCHS = 200
LR = 3e-4  
TOLERENCE= 1e-1
BATCH_SIZE_TWO = 32

import pandas as pd
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import re
import string
from torchinfo import summary
import torch.optim as optim
from torchtext.legacy import data
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.model_selection import train_test_split


# In[3]:


'''loading the pretrained embedding weights'''
weights=torch.load('CBOW_NEWS.pth')
pre_trained = nn.Embedding.from_pretrained(weights)
pre_trained.weight.requires_grad=False


# In[4]:


def collate_batch(batch):
    label_list, text_list, length_list = [], [], []
    for (_text,_label, _len) in batch:
        label_list.append(_label)
        length_list.append(_len)
        tensor = torch.tensor(_text, dtype=torch.long)
        text_list.append(tensor)
    text_list = pad_sequence(text_list, batch_first=True)
    label_list = torch.tensor(label_list, dtype=torch.float)
    length_list = torch.tensor(length_list)
    return text_list,label_list, length_list

class VectorizeData(Dataset):
    def __init__(self, file):
        self.data = pd.read_pickle(file)
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        X = self.data.vector[idx]
        lens = self.data.lengths[idx]
        y = self.data.label[idx]
        return X,y,lens
    
    
training = VectorizeData('variable_level_zero.csv') 
dt_load = DataLoader(training, batch_size=BATCH_SIZE_TWO, shuffle=False, collate_fn=collate_batch)


# In[140]:


def binary_accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc

def create_emb_layer(pre_trained):
    num_embeddings = pre_trained.num_embeddings
    embedding_dim = pre_trained.embedding_dim
    emb_layer = nn.Embedding.from_pretrained(pre_trained.weight.data, freeze=True)
    return emb_layer, embedding_dim

class C_DNN(nn.Module):
    def __init__(self, pre_trained,num_labels):
        super(C_DNN, self).__init__()
        self.n_class = num_labels
        self.embedding, self.embedding_dim = create_emb_layer(pre_trained)
        self.conv1D = nn.Conv2d(1, 100, kernel_size=(3,16), padding=(1,0))
        self.label = nn.Linear(100, self.n_class)
        self.act = nn.Sigmoid()
        
    def forward(self, x):
        embeds = self.embedding(x)
        embeds = embeds.unsqueeze(1)
        conv1d = self.conv1D(embeds)
        relu = F.relu(conv1d).squeeze(3)
        maxpool = F.max_pool1d(input=relu, kernel_size=relu.size(2)).squeeze(2)
        fc = self.label(maxpool)
        sig = self.act(fc)
        return sig.squeeze(1)


# In[145]:


model = C_DNN(pre_trained=pre_trained, num_labels=1)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCELoss()

def train(dataloader, model, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    steps = 0
    model.train() 
    for idx, batch in enumerate(dataloader): 
        text,label,lengths = batch
        optimizer.zero_grad() 
        prediction = model(text)
        loss = criterion(prediction, label) 
        acc = binary_accuracy(prediction, label)
        #backpropage the loss and compute the gradients
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        #update the weights
        optimizer.step()  
        steps += 1
        if steps % 50 == 0:
            print(f'Epoch: {epoch}, Idx: {idx+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item():.2f}%')
        total_epoch_loss = loss.item()
        if total_epoch_loss <= TOLERENCE:
            return

filename = "models/model_"
for epoch in range(1, EPOCHS + 1):
    train(dt_load, model, epoch)
filename = "models/model_"+str(3)+'.pth'
torch.save(model.state_dict(), filename)

