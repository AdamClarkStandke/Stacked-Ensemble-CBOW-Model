#!/usr/bin/env python
# coding: utf-8

# In[ ]:


EPOCHS = 40
LR = 3e-4  
BATCH_SIZE_TWO = 1
HIDDEN =20
MEMBERS = 3

import pandas as pd
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchinfo import summary
import re
import string
import torch.optim as optim
from torchtext.legacy import data
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.model_selection import train_test_split


# In[ ]:


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
    
testing = VectorizeData('variable_test_set.csv')
dtes_load = DataLoader(testing, batch_size=BATCH_SIZE_TWO, shuffle=False, collate_fn=collate_batch)


# In[ ]:


'''loading the pretrained embedding weights'''
weights=torch.load('CBOW_NEWS.pth')
pre_trained = nn.Embedding.from_pretrained(weights)
pre_trained.weight.requires_grad=False


# In[ ]:


# Part Implemntation of Integrated Stacking Model as detailed by 
# Jason Brownlee 
# @ https://machinelearningmastery.com/stacking-ensemble-for-deep-learning
# -neural-networks/

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

class StackedLSTMAtteionModel(nn.Module):
    def __init__(self, pre_trained,num_labels):
        super(StackedLSTMAtteionModel, self).__init__()
        self.n_class = num_labels
        self.embedding, self.embedding_dim = create_emb_layer(pre_trained)
        self.LSTM = nn.LSTM(self.embedding_dim, HIDDEN, num_layers=2,bidirectional=True,dropout=0.26,batch_first=True)
        self.label = nn.Linear(2*HIDDEN, self.n_class)
        self.act = nn.Sigmoid()
        
    def attention_net(self, Lstm_output, final_state):
        hidden = final_state
        output = Lstm_output[0]
        attn_weights = torch.matmul(output, hidden.transpose(1, 0))
        soft_attn_weights = F.softmax(attn_weights.transpose(1, 0), dim=1)
        new_hidden_state = torch.matmul(output.transpose(1,0), soft_attn_weights.transpose(1,0))
        return new_hidden_state.transpose(1, 0)
    
    def forward(self, x, text_len):
        embeds = self.embedding(x)
        pack = pack_padded_sequence(embeds, text_len, batch_first=True, enforce_sorted=False)
        output, (hidden, cell) = self.LSTM(pack)
        hidden = torch.cat((hidden[0,:, :], hidden[1,:, :]), dim=1)
        attn_output = self.attention_net(output, hidden)
        logits = self.label(attn_output)
        outputs = self.act(logits.view(-1))
        return outputs
    
    
class TwoLayerGRUAttModel(nn.Module):
    def __init__(self, pre_trained, HIDDEN, num_labels):
        super(TwoLayerGRUAttModel, self).__init__()
        self.n_class = num_labels
        self.embedding, self.embedding_dim = create_emb_layer(pre_trained)
        self.gru = nn.GRU(self.embedding_dim, hidden_size=HIDDEN, num_layers=2,batch_first=True, bidirectional=True, dropout=0.2)
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
    
class MetaLearner(nn.Module):
    def __init__(self, modelA, modelB, modelC):
        super(MetaLearner, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.fc1 = nn.Linear(3, 2)
        self.fc2 = nn.Linear(2, 1)
        self.act = nn.Sigmoid()
    
    def forward(self, text, length):
        x1=self.modelA(text, length) 
        x2=self.modelB(text,length)
        x3=self.modelC(text)
        x4 = torch.cat((x1.detach(),x2.detach(), x3.detach()), dim=0)
        x5 = F.relu(self.fc1(x4))
        output = self.act(self.fc2(x5))
        return output


# In[ ]:


def load_all_models(n_models):
    all_models = []
    for i in range(n_models):
        filename = "models/model_"+str(i+1)+'.pth'
        if filename == "models/model_1.pth": 
            model_one = StackedLSTMAtteionModel(pre_trained, 1)
            model_one.load_state_dict(torch.load(filename))
            for param in model_one.parameters():
                param.requires_grad = False
            all_models.append(model_one)
        elif filename == "models/model_2.pth":
            model_two = TwoLayerGRUAttModel(pre_trained, HIDDEN, 1)
            model_two.load_state_dict(torch.load(filename))
            for param in model_two.parameters():
                param.requires_grad = False
            all_models.append(model_two)
        else:
            model = C_DNN(pre_trained=pre_trained, num_labels=1)
            model.load_state_dict(torch.load(filename))
            for param in model.parameters():
                param.requires_grad = False
            all_models.append(model)
    return all_models


# In[ ]:


models = load_all_models(MEMBERS)
meta_model = MetaLearner(models[0], models[1], models[2])
optimizer = optim.Adam(meta_model.parameters(), lr=LR)
criterion = nn.BCELoss()

def validate_meta(dataloader, model, epoch):
    #initialize every epoch 
    total_epoch_loss = 0
    total_epoch_acc = 0
    #set the model in training phase
    model.train()
    for idx, batch in enumerate(dataloader): 
        text,label,lengths = batch 
        optimizer.zero_grad() 
        prediction = model(text, lengths) 
        loss = criterion(prediction, label)
        #backpropage the loss and compute the gradients
        loss.backward()  
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        #update the weights
        optimizer.step() 
        total_epoch_loss += loss.item()
        if idx % 1 == 0 and idx > 0:
            print(f'Epoch: {epoch}, Idx: {idx}, Meta Training Loss: {total_epoch_loss:.4f}')

for epoch in range(1, EPOCHS + 1):
    validate_meta(dtes_load, meta_model, epoch)
filename = "models/model_metaLearner_two.pth"
torch.save(meta_model.state_dict(), filename)


# In[ ]:




