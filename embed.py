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
        indexs=np.zeros(shape=(len(x),),dtype=np.int32)
        for i in range(len(indexs)):
            indexs[i]=self.nn(x[i])
        return indexs
    def nn(self,vector):
        vector=vector.numpy()
        errors=self.processor.vectors-vector
        errors**=2 #produce squared error among all vectors
        index=np.argmin(errors.mean(axis=1))
        return index
"""
b=BpEmbed(hparams,bpemb_en)
t=b(torch.tensor([0,1,20,3333,444,5,88]))
#.2 noise scale is save, .3 pushes it, .4 causes errors .1 for little ambiguity
print(t)
print()
print()
for k in range(1000):
    print(k)
    noise=torch.tensor(np.random.random_sample(t.shape)).float()*.30
    a=b.disembed(t)
    c=b.disembed(t+noise)
    if(not np.array_equal(a,c)):
        print("Mismatch!!!")
"""
