#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


EPOCHS = 30
LR = 3e-4  
HIDDEN = 20
TOLERENCE= 1e-1
BATCH_SIZE_TWO = 32

import pandas as pd
import numpy as np
import random
import torch
from torchinfo import summary
import torch.nn.functional as F
import torch.nn as nn
import re
import string
import torch.optim as optim
from torchtext.legacy import data
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.model_selection import train_test_split


# In[3]:


# word2idx = torch.load('dictionary/wordDict.pth')

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


# In[5]:


# Part implementation of aravindpal's Text Classifier as found @  https://www.analyticsvidhya.com/blog/2020/01/first-text-classification-in-pytorch/
# and Prakash Pandey's LSTM+Attention model as found @ https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/models/LSTM_Attn.py

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

class EmbeddingGRUAtteionModel(nn.Module):
    def __init__(self, pre_trained,num_labels):
        super(EmbeddingGRUAtteionModel, self).__init__()
        self.n_class = num_labels
        self.embedding, self.embedding_dim = create_emb_layer(pre_trained)
        self.gru = nn.GRU(self.embedding_dim, HIDDEN, num_layers=2,batch_first=True, bidirectional=True, dropout=0.2)
        self.label = nn.Linear(2*HIDDEN, self.n_class)
        self.act = nn.Sigmoid()
        
    def attention_net(self, gru_output, final_state):
        hidden = final_state
        output = gru_output[0]
        attn_weights = torch.matmul(output, hidden.transpose(1, 0))
        soft_attn_weights = F.softmax(attn_weights.transpose(1, 0), dim=1)
        new_hidden_state = torch.matmul(output.transpose(1,0), soft_attn_weights.transpose(1,0))
        return new_hidden_state.transpose(1, 0)
    
    def forward(self, x, text_len):
        embeds = self.embedding(x)
        pack = pack_padded_sequence(embeds, text_len, batch_first=True, enforce_sorted=False)
        output, hidden = self.gru(pack)
        hidden = torch.cat((hidden[0,:, :], hidden[1,:, :]), dim=1)
        attn_output = self.attention_net(output, hidden)
        logits = self.label(attn_output)
        outputs = self.act(logits.view(-1))
        return outputs


# In[6]:


model = EmbeddingGRUAtteionModel(pre_trained=pre_trained, num_labels=1)
optimizer = optim.Adam(model.parameters())
criterion = nn.BCELoss()


# In[ ]:


def train(dataloader, model, epoch):
    #initialize every epoch 
    total_epoch_loss = 0
    total_epoch_acc = 0
    steps = 0
    #set the model in training phase
    model.train()  
    for idx, batch in enumerate(dataloader): 
        text,label,lengths = batch
        optimizer.zero_grad() 
        prediction = model(text, lengths)
        loss = criterion(prediction, label) 
        acc = binary_accuracy(prediction, label)
        #backpropage the loss and compute the gradients
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        #update the weights
        optimizer.step()  
        steps += 1
        if steps % 1 == 0:
            print(f'Epoch: {epoch}, Idx: {idx+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item():.2f}%')
        total_epoch_loss = loss.item()
        if total_epoch_loss <= TOLERENCE:
            return

filename = "models/model_"
for epoch in range(1, EPOCHS + 1):
    train(dt_load, model, epoch)
filename = "models/model_"+str(2)+'.pth'
torch.save(model.state_dict(), filename)


# In[ ]:




