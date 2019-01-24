import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from hparams import hparams

class Pos_Encoding_Like(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        pe = Parameter(torch.zeros(1,hparams['dim'],hparams['batch_size']),requires_grad=False)
        for p in range(hparams['batch_size']):
            for i in range(0, params['dim'], 2):
                pe[0,i,p]=math.sin(p / (10000 ** ((2 * i)/hparams['dim'])))
                pe[0,i+1,p]=math.cos(p / (10000 ** ((2 * i)/hparams['dim'])))
        pe/=params['dim']**1/2
    def forward(self,x):
        shape=x.shape
        x=self.pe[:,:,0:shape[2]]
        return x
class Learned_Encoding_Like(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        self.pe = Parameter(torch.zeros(1,hparams['dim'],hparams['batch_size']))
    def forward(self,x):
        shape=x.shape
        x=self.pe[:,:,0:shape[2]]
        return x
class Conv1d(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        self.linweight=Parameter(torch.empty(size=(hparams['dim'],hparams['dim'])))
        torch.nn.init.orthogonal_(self.linweight) #orthogonal
        #self.register_parameter("w",self.linweight)
        self.bias=Parameter(torch.zeros(size=(1,hparams['dim'],1,)))
        #self.register_parameter("b",self.bias)
    def forward(self,x):
        return F.conv1d(x,torch.unsqueeze(self.linweight,-1))+self.bias
    def inverse(self,x):
        x=x-self.bias
        invlin=F.conv1d(x,torch.unsqueeze(torch.inverse(self.linweight),-1))
        logdet=torch.slogdet(self.linweight)[1]#/self.hparams['dim']
        #print("Conv logdet " + str(logdet))
        return (invlin,logdet)
    def cuda(self):
        self.bias=self.bias.cuda
        self.linweight=self.linweight.cuda()
        print("hi")
class Conv_block(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        self.act=torch.nn.ReLU()
        self.conva=torch.nn.Conv1d(hparams['dim']//2,hparams['dim']//2,3,padding=1)
        self.convb=torch.nn.Conv1d(hparams['dim']//2,hparams['dim']//2,3,padding=1)
        self.multiply_conv=torch.nn.Conv1d(hparams['dim']//2,hparams['dim']//2,1)
        self.add_conv=torch.nn.Conv1d(hparams['dim']//2,hparams['dim']//2,1)
        self.mscale=nn.Parameter(torch.zeros(size=(1,hparams['dim']//2,1)))
        self.ascale=nn.Parameter(torch.zeros(size=(1,hparams['dim']//2,1)))
    def forward(self,x):
        x=self.conva(x)
        x=self.act(x)
        x=self.convb(x)
        x=self.act(x)
        m=self.multiply_conv(x)*self.mscale
        a=self.add_conv(x)#*self.ascale
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
class Identity(nn.Module):
    def __init__(self,hparams):
        super().__init__()
    def forward(self,x):
        return x
    def inverse(self,x):
        return (x,torch.tensor(0))
class FC_block(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        self.lin=Conv1d(hparams)#Conv1d(hparams)
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
        self.encoding_like=Learned_Encoding_Like(hparams)
        blocks=[FC_block(hparams) for i in range(hparams['blocks'])] #FC_block
        blocks.append(Conv1d(hparams))
        #print(blocks)
        self.blocks=nn.ModuleList(blocks)
    def forward(self,x):
        #x+=self.encoding_like(x)
        for block in self.blocks:
            x=block(x)
        #x-=self.encoding_like(x)
        return x
    def inverse(self,x):
        #x+=self.encoding_like(x)
        state=[x,0] #current inverse, log determinant
        for block in reversed(self.blocks):
            inv=block.inverse(state[0])
            state[0]=inv[0]
            state[1]=state[1]+inv[1]
        #state[0]-=self.encoding_like(x)
        return state
