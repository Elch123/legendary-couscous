import torch
import torch.nn as nn
import torch.nn.functional as F
from hparams import hparams

class Lin_bidirectional(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        self.linweight=nn.Parameter(torch.empty(size=(hparams['channels'],hparams['channels'])))
        torch.nn.init.orthogonal_(self.linweight) #orthogonal
        self.register_parameter("w",self.linweight)
        self.bias=nn.Parameter(torch.zeros(size=(hparams['channels'],)))
        self.register_parameter("b",self.bias)
    def forward(self,x):
        return F.linear(x,self.linweight)+self.bias
    def inverse(self,x):
        x=x-self.bias
        invlin=F.linear(x,torch.inverse(self.linweight))
        logdet=torch.slogdet(self.linweight)[1]#/self.hparams['channels']
        #print("Conv logdet " + str(logdet))
        return (invlin,logdet)
class Res_block(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        self.act=torch.nn.ReLU()
        self.lina=torch.nn.Linear(hparams['channels']//2,hparams['channels']//2)
        self.linb=torch.nn.Linear(hparams['channels']//2,hparams['channels']//2)
    def forward(self,x):
        prex=x
        x=self.act(x)
        x=self.lina(x)
        x=self.act(x)
        x=self.linb(x)
        x+=prex
        return x

class Affine(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        self.act=torch.nn.ReLU()
        self.lina=torch.nn.Linear(hparams['channels']//2,hparams['channels']//2)
        self.linb=torch.nn.Linear(hparams['channels']//2,hparams['channels']//2)
        self.linc=torch.nn.Linear(hparams['channels']//2,hparams['channels']//2)
        self.lind=torch.nn.Linear(hparams['channels']//2,hparams['channels']//2)
        self.mults=torch.nn.Linear(hparams['channels']//2,hparams['channels']//2)
        self.adds=torch.nn.Linear(hparams['channels']//2,hparams['channels']//2)
        self.ms=nn.Parameter(torch.zeros(size=(hparams['channels']//2,)))
    def compute_nets(self,x):
        prex=x
        x=self.act(x)
        x=self.lina(x)
        x=self.act(x)
        x=self.linb(x)
        x+=prex
        prex=x
        x=self.act(x)
        x=self.linc(x)
        x=self.act(x)
        x=self.lind(x)
        x+=prex
        x=self.act(x)
        m=self.mults(x)*self.ms
        a=self.adds(x)
        return (m,a)
    def forward(self,data):
        x=data[:,0:self.hparams['channels']//2]
        y=data[:,self.hparams['channels']//2:self.hparams['channels']]
        (m,a)=self.compute_nets(x)
        data=torch.cat((x,y*torch.exp(m)+a),dim=1)
        return data
    def inverse(self,data):
        x=data[:,0:self.hparams['channels']//2]
        (m,a)=self.compute_nets(x)
        y=data[:,self.hparams['channels']//2:self.hparams['channels']]
        data=torch.cat((x,(y-a)/torch.exp(m)),dim=1)
        logdet=torch.sum(m,dim=1)
        return (data,logdet)

class FC_block(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        self.lin=Lin_bidirectional(hparams)
        self.act=Affine(hparams)#Parametric_Affine
    def forward(self,x):
        return self.act(self.lin(x))
    def inverse(self,x):
        postconv=self.act.inverse(x)
        start=self.lin.inverse(postconv[0])
        logdet=postconv[1]+start[1]
        return (start[0],logdet)
class Net(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        blocks=[FC_block(hparams) for i in range(hparams['blocks'])] #FC_block
        blocks.append(Lin_bidirectional(hparams))
        #print(blocks)
        self.blocks=nn.ModuleList(blocks)
    def forward(self,x):
        for block in self.blocks:
            x=block(x)
        return x
    def inverse(self,x):
        state=[x,torch.tensor(0.0,requires_grad=True)] #current inverse, log determinant
        for block in reversed(self.blocks):
            inv=block.inverse(state[0])
            state[0]=inv[0]
            state[1]=state[1]+inv[1]
        return state
