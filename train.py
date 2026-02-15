import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
#import torchvision.transforms as transforms
#import numpy as np
from tqdm import tqdm
import pickle
import time
#import subprocess
from transformerlayer import TransformerLayer
import sentencepiece as sp
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler

sp = sp.SentencePieceProcessor(model_file='sentence_piece_tok.model')



lrate = 0.0005
bs = 32
count_ep_for_save = 1000
layers_count = 20
epo = 1




mpllist = []


try:
    with open("index_l.pkl", "rb") as f:
        index_l = pickle.load(f)

    with open("next_tok_l.pkl", "rb") as f:
        next_tok_l = pickle.load(f)

    with open("q_list.pkl", "rb") as f:
        q_list = pickle.load(f)

    with open("a_list.pkl", "rb") as f:
        a_list = pickle.load(f)

    with open("numsym_l.pkl", "rb") as f:
        numsym_l = pickle.load(f)
except:
    print('отсутствуют необходимые файлы')

gllent = int(0)
lencount = int(0)






class LLM_Dataset(Dataset):
    def __init__(self):
        self.index_l = index_l
        self.next_tok_l = next_tok_l
        self.q_list = q_list
        self.a_list = a_list
        self.numsym_l = numsym_l


    def __len__(self):
        return len(self.next_tok_l)#[:500000]

    def __getitem__(self, idx):
        list1 = self.q_list[self.index_l[idx]].copy()
        list1 += self.a_list[self.index_l[idx]][:self.numsym_l[idx]]
        next_tok = self.next_tok_l[idx]
        

        global gllent
        global lencount

        #print(sp.decode(list1))
        #print('--------------')
        #print(sp.decode(next_tok))
        #time.sleep(1)
        #print('--------------')
        #print('--------------')
        #print('--------------')
        gllent += (len(list1))
        lencount += 1

        if len(list1) > 256:
            list1 = list1[-256:]
        else:
            list1 = list1 + [0] * (256 - len(list1))



        next_tok = list1[1:] + [next_tok]


        list1 = torch.tensor(list1, dtype=torch.long)
        next_tok = torch.tensor(next_tok, dtype=torch.long)

        return list1, next_tok






class llm(nn.Module):
    def __init__(self):
        global layers_count
        super().__init__()
        self.embed = nn.Embedding(512, 256)
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim=256, ff_dim=512, n_heads=4, dropout_rate=0.06)
            for _ in range(layers_count)
        ])
        self.log = nn.Linear(256, 512)
        self.pos_embed = nn.Embedding(256, 256)  # 128 - SL, 128 - embed_dim


    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).expand(B, -1)
        x = self.embed(x) + self.pos_embed(pos)
        for layer in self.layers:
            x = layer(x)
        x = self.log(x)
        return x









device = torch.device('cuda')

model = llm()

model.to(device)


dataset = LLM_Dataset()
dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)

try:
    model.load_state_dict(torch.load("model.pth"))
    print('найдена модель начало дообучения существующей модели...')
except:
    print('модель не найдена создание новой модели...')


optimizer = optim.Adam(model.parameters(), lr=lrate, weight_decay=0.01)
#optimizer = optim.Adam(model.parameters(), lr = lrate)
loss_func = nn.CrossEntropyLoss(ignore_index=0)




#with autocast(dtype=torch.float16):
for epoch in range(epo):
    model.train()
    total_loss = 0.0
    batches = 0
    ex = 0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))

    for index, (list1, next_tok) in pbar:

        list1 = list1.to(device)
        next_tok = next_tok.to(device)

        optimizer.zero_grad()#!!!!


        output = model(list1)

        loss = loss_func(output.view(-1, 512), next_tok.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        batches += 1

        del output, loss
        if index % 100 == 0 and not index == 0.0:
            avg_loss = total_loss / batches
            lossstr = (f"loss: {avg_loss:.4f}")
            pbar.set_postfix_str(lossstr)
            mpllist.append((round(avg_loss, 4)))
            total_loss = 0.0


        if index % count_ep_for_save == 0:
            torch.save(model.state_dict(), "model.pth")


    torch.save(model.state_dict(), "fin_model.pth")
print("все")

plt.plot(mpllist)
plt.savefig('loss.png', dpi=300, bbox_inches='tight')


print(f"средняя длинна: {gllent / lencount}")
