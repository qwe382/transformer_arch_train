import os
#import torch
#import torch.nn as nn
#import torch.optim as optim
#from torch.utils.data import Dataset, DataLoader
#import numpy as np
import json
from tqdm import tqdm
#from transformer import transformerlayer
import re
import sentencepiece as sp
import pickle
#import time


sp = sp.SentencePieceProcessor(model_file='sentence_piece_tok.model')

q_list = []
a_list = []


with open("data.json", "r", encoding="utf-8") as f:
    for line in (tqdm(f, desc="чтение файла")):
        line = line.strip()
        line = json.loads(line)
        q = line['q']
        a = line['a']
        if not any(c.isdigit() for c in q) and not any(c.isdigit() for c in a):
            q = re.sub(r'[^a-zA-Z0-9 \n]', '', q)
            a = re.sub(r'[^a-zA-Z0-9 \n]', '', a)
            q = q.lower()
            a = a.lower()
            q = sp.encode(q, out_type=int)
            a = sp.encode(a, out_type=int)
            #print(q)
            #print(a)
            q.append(1)
            a.append(2)
            q_list.append(q)
            a_list.append(a)
            #print('q:', sequence[-1], "a:", a_list[-1])



    #print(sp.decode(q_list[-1]),'next tok:', sp.decode(a_list[-1]))
       
    f.close()
    del f, a, q, line

    with open("q_list.pkl", "wb") as f:
        pickle.dump(q_list, f)

    with open("a_list.pkl", "wb") as f:
        pickle.dump(a_list, f)


context_l = []
next_tok_l = []



#print(sp.decode(q_list[0]))
#print(sp.decode(a_list[0]))

index_l = []
numsym_l = []

for index, answr in enumerate(tqdm(a_list)):
    next_token_loc = []
    num = 0
    #print("answr=", answr)
    for idx, i in enumerate(answr):
        #list1 = q_list[index].copy()
        #list1 += a_list[index][:num]
        
        index_l.append(index)
        numsym_l.append(num)
        
        next_token_loc = (a_list[index][idx])
        num += 1
        #print(sp.decode(list1))
        #print(sp.decode(next_token_loc))
        #print(len(context_l),len(fin_nt_l))
        #time.sleep(0.1)
        
        
        
        
        #context_l.append(list1)
        next_tok_l.append(next_token_loc)

    #print(sp.decode(context_l[-1][-1]))
    #print('')
    #print(sp.decode(fin_nt_l[-1][-1]))
    #time.sleep(1)
    #print('')
    #print('')
    #print('')
    #print('')

with open("index_l.pkl", "wb") as f:
    pickle.dump(index_l, f)

with open("numsym_l.pkl", "wb") as f:
    pickle.dump(numsym_l, f)

with open("next_tok_l.pkl", "wb") as f:
    pickle.dump(next_tok_l, f)










