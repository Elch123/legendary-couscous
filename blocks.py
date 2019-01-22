import torch
import torch.nn as nn
import torch.nn.functional as F
from hparams import hparams


class Conv1d(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        self.linweight=nn.Parameter(torch.empty(size=(hparams['dim'],hparams['dim'])))
        torch.nn.init.orthogonal_(self.linweight) #orthogonal
        self.register_parameter("w",self.linweight)
        self.bias=nn.Parameter(torch.zeros(size=(1,hparams['dim'],1,)))
        self.register_parameter("b",self.bias)
    def forward(self,x):
        return F.conv1d(x,torch.unsqueeze(self.linweight,-1))+self.bias
    def inverse(self,x):
        x=x-self.bias
        invlin=F.conv1d(x,torch.unsqueeze(torch.inverse(self.linweight),-1))
        logdet=torch.slogdet(self.linweight)[1]#/self.hparams['dim']
        #print("Conv logdet " + str(logdet))
        return (invlin,logdet)
class Conv_block(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        self.act=torch.nn.ReLU()
        self.conva=torch.nn.Conv1d(hparams['dim']//2,hparams['dim']//2,3,padding=1)
        self.convb=torch.nn.Conv1d(hparams['dim']//2,hparams['dim']//2,3,padding=1)
        self.multiply_conv=torch.nn.Conv1d(hparams['dim']//2,hparams['dim']//2,1)
        self.add_conv=torch.nn.Conv1d(hparams['dim']//2,hparams['dim']//2,1)
        self.ms=nn.Parameter(torch.zeros(size=(1,hparams['dim']//2,1)))
    def forward(self,x):
        x=self.conva(x)
        x=self.act(x)
        x=self.convb(x)
        x=self.act(x)
        m=self.multiply_conv(x)*self.ms
        a=self.add_conv(x)
        return (m,a)
class Affine1d(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        self.block=Conv_block(hparams)
    def forward(self,data):
        x=data[:,0:self.hparams['dim']//2,:]
        y=data[:,self.hparams['dim']//2:self.hparams['dim'],:]
        (m,a)=self.block(x)
        out=torch.cat((x,y*torch.exp(m)+a),dim=1)
        return out
    def inverse(self,data):
        x=data[:,0:self.hparams['dim']//2,:]
        (m,a)=self.block(x)
        y=data[:,self.hparams['dim']//2:self.hparams['dim'],:]
        data=torch.cat((x,(y-a)/torch.exp(m)),dim=1)
        logdet=torch.sum(m,dim=1)
        return (data,logdet)
class Affine(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        self.act=torch.nn.ReLU()
        self.lina=torch.nn.Linear(hparams['dim']//2,hparams['dim']//2)
        self.linb=torch.nn.Linear(hparams['dim']//2,hparams['dim']//2)
        self.linc=torch.nn.Linear(hparams['dim']//2,hparams['dim']//2)
        self.lind=torch.nn.Linear(hparams['dim']//2,hparams['dim']//2)
        self.mults=torch.nn.Linear(hparams['dim']//2,hparams['dim']//2)
        self.adds=torch.nn.Linear(hparams['dim']//2,hparams['dim']//2)
        self.ms=nn.Parameter(torch.zeros(size=(hparams['dim']//2,)))
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
        x=data[:,0:self.hparams['dim']//2]
        y=data[:,self.hparams['dim']//2:self.hparams['dim']]
        (m,a)=self.compute_nets(x)
        data=torch.cat((x,y*torch.exp(m)+a),dim=1)
        return data
    def inverse(self,data):
        x=data[:,0:self.hparams['dim']//2]
        (m,a)=self.compute_nets(x)
        y=data[:,self.hparams['dim']//2:self.hparams['dim']]
        data=torch.cat((x,(y-a)/torch.exp(m)),dim=1)
        logdet=torch.sum(m,dim=1)
        return (data,logdet)

class FC_block(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        self.lin=Conv1d(hparams)
        self.act=Affine1d(hparams)#Parametric_Affine
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
        blocks.append(Conv1d(hparams))
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
