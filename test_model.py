import os
import torch
import torch.nn as nn
from transformerlayer import TransformerLayer
import sentencepiece as sp

sp = sp.SentencePieceProcessor(model_file='sentence_piece_tok.model')
layers_count = 20


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
model = llm().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

inpt = input("введите запрос: ")
inpt = sp.encode(inpt, out_type=int)
inpt.append(1)

print("Вход:", sp.decode(inpt))
print("Длина:", len(inpt))

a = []
for i in range(120):
    if len(inpt) > 128:
        inpt = inpt[-128:]
    #else:
    #    inpt = inpt + [0] * (128 - len(inpt))





    x = torch.tensor(inpt, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        nt = model(x)
        ntid = torch.argmax(nt[:, -1, :], dim=-1).item()

    if ntid == 2:
        break

    inpt.append(ntid)
    a.append(ntid)

print("Ответ:", sp.decode(a))
