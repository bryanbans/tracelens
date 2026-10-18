"""Word-level multiple-instance CNN: window labels, content-only features."""
import copy
import re
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

WORDS=re.compile(r'<[^>]+>|[a-z]+|[+_-]')
TOKENS_PER_EVENT=16


def words(template):
    return WORDS.findall(template.lower())[:TOKENS_PER_EVENT]


def word_vocabulary(groups):
    count=Counter(w for group in groups for event in group for w in words(event['template']))
    return {'<pad>':0,'<unk>':1,**{w:i+2 for i,w in enumerate(sorted(w for w,n in count.items() if n>=2 and w not in {'<pad>','<unk>'}))}}


def encode_groups(groups, vocabulary):
    result=np.zeros((len(groups),len(groups[0]),TOKENS_PER_EVENT),dtype=np.int64)
    for i,group in enumerate(groups):
        for j,event in enumerate(group):
            tokens=[vocabulary.get(w,1) for w in words(event['template'])]
            result[i,j,:len(tokens)]=tokens
    return result


class TextCNN(nn.Module):
    def __init__(self,vocab_size,embedding=24,channels=24,dropout=.15):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embedding,padding_idx=0)
        self.convs=nn.ModuleList([nn.Conv1d(embedding,channels,k,padding=k//2) for k in (3,5)])
        self.head=nn.Sequential(nn.Linear(channels*2,32),nn.GELU(),nn.Dropout(dropout),nn.Linear(32,1))

    def event_logits(self,x):
        b,e,t=x.shape
        features=self.embedding(x.reshape(b*e,t)).transpose(1,2)
        valid=x.reshape(b*e,t)!=0
        pooled=[]
        for conv in self.convs:
            value=F.gelu(conv(features)).masked_fill(~valid[:,None,:],-1e4)
            pooled.append(value.amax(dim=2))
        return self.head(torch.cat(pooled,dim=1)).reshape(b,e)

    def forward(self,x):
        return self.event_logits(x).amax(dim=1)


def score_text(model,x,batch_size=128):
    model.eval(); parts=[]
    with torch.inference_mode():
        for start in range(0,len(x),batch_size):
            parts.append(model.event_logits(torch.as_tensor(x[start:start+batch_size])).sigmoid().numpy())
    event_scores=np.concatenate(parts)
    return event_scores.max(axis=1).astype(float),event_scores.astype(float)


def train_text(train_x,train_y,val_x,val_y,vocab_size,seed,epochs=6):
    torch.manual_seed(seed);np.random.seed(seed);torch.use_deterministic_algorithms(True)
    net=TextCNN(vocab_size);optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    positive_weight=torch.tensor(float((train_y==0).sum()/max(1,(train_y==1).sum())))
    loader=DataLoader(TensorDataset(torch.as_tensor(train_x),torch.as_tensor(train_y,dtype=torch.float32)),batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(seed))
    best=float('inf');saved=None;history=[];stale=0
    for epoch in range(epochs):
        net.train();total=0
        for x,y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss=F.binary_cross_entropy_with_logits(net(x),y,pos_weight=positive_weight)
            loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1);optimizer.step();total+=loss.item()*len(x)
        net.eval();val_losses=[]
        with torch.inference_mode():
            for start in range(0,len(val_x),128):
                xb=torch.as_tensor(val_x[start:start+128]);yb=torch.as_tensor(val_y[start:start+128],dtype=torch.float32)
                val_losses.append((F.binary_cross_entropy_with_logits(net(xb),yb,pos_weight=positive_weight).item(),len(xb)))
        val_loss=sum(a*b for a,b in val_losses)/len(val_x)
        history.append({'epoch':epoch+1,'train_loss':total/len(train_x),'validation_loss':val_loss})
        print(f'semantic seed={seed} epoch={epoch+1} train={total/len(train_x):.4f} validation={val_loss:.4f}',flush=True)
        if val_loss<best:
            best=val_loss;saved=copy.deepcopy(net.state_dict());stale=0
        else:stale+=1
        if stale>=2:break
    net.load_state_dict(saved)
    return net,history
