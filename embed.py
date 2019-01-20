from loader import makeendeprocessors
(bpemb_en,bpemb_de)=makeendeprocessors()
import torch
import torch.nn as nn
from hparams import hparams
import numpy as np
class BpEmbed(nn.Module):
    def __init__(self,hparams,processor):
        super().__init__()
        self.hparams=hparams
        self.processor=processor
        self.embed=torch.nn.Embedding(hparams['embed_symbols'],hparams['dim'],_weight=torch.tensor(processor.vectors))
        for param in self.embed.parameters():
            param.requires_grad=False
    def forward(self,x):
        return self.embed(x)
    def disembed(self,x):
        indexs=np.zeros(shape=(len(x),))
        for i in range(len(indexs)):
            indexs[i]=self.nn(x[i])
        return indexs
    def nn(self,vector):
        vector=vector.numpy()
        errors=self.processor.vectors-vector
        errors**=2 #produce squared error among all vectors
        index=np.argmin(errors.mean(axis=1))
        return index
b=BpEmbed(hparams,bpemb_en)
t=b(torch.tensor([0,1,2,3,4,5,88]))
print(t)
print(b.disembed(t))
