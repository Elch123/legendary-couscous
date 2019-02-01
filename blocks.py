import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from hparams import hparams
import tracemalloc
import math

class Conv1d(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        self.linweight=Parameter(torch.empty(size=(hparams['dim'],hparams['dim'],1)))
        torch.nn.init.orthogonal_(self.linweight) #orthogonal
        #self.register_parameter("w",self.linweight)
        self.bias=Parameter(torch.zeros(size=(1,hparams['dim'],1,)))
        #self.register_parameter("b",self.bias)
    def forward(self,x):
        return F.conv1d(x,torch.unsqueeze(torch.inverse(torch.squeeze(self.linweight)),-1))+self.bias
    def inverse(self,x):
        x=x-self.bias
        invlin=F.conv1d(x,self.linweight)
        logdet=-torch.slogdet(torch.squeeze(self.linweight))[1]#/self.hparams['dim']
        #print("Conv logdet " + str(logdet))
        return (invlin,logdet)
class Conv1d_broken(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
         #orthogonal
        self.inv_conv=nn.Conv1d(hparams['dim'],hparams['dim'],1,bias=False)
        self.forward_conv=nn.Conv1d(hparams['dim'],hparams['dim'],1,bias=False)
        linweight=torch.empty(size=(hparams['dim'],hparams['dim']))
        torch.nn.init.orthogonal_(linweight)
        self.inv_conv.weight.data=torch.unsqueeze(linweight,-1)
        self.bias=Parameter(torch.zeros(size=(1,hparams['dim'],1,)))
        self.dirty=True
    def forward(self,x):
        if(self.dirty):
            self.dirty=False
            inverse_kernel=torch.inverse(torch.squeeze(self.inv_conv.weight.data))
            self.forward_conv.weight.data=torch.unsqueeze(inverse_kernel,-1)
        x=self.forward_conv(x)
        x=x+self.bias
        return  x
    def inverse(self,x):
        x=x-self.bias
        self.dirty=True
        invlin=self.inv_conv(x)
        logdet=-torch.slogdet(torch.squeeze(self.inv_conv.weight.data))[1]#The log determinant of the inverse matrix is the negative of the forward one.
        #logdet=0
        return (invlin,logdet)
class Pos_Encoding_Like(nn.Module): #slow,perhaps optimize, but only needs one run
    def __init__(self,hparams,embedding_channels):
        super().__init__()
        self.hparams=hparams
        d=embedding_channels
        if(d%2==1):
            d=d+1
        pe = torch.zeros(1,d,hparams['batch_size'])
        for p in range(hparams['batch_size']):
            for i in range(0, d, 2):
                pe[0,i,p]=math.sin(p / (10000 ** ((2 * i)/d)))
                pe[0,i+1,p]=math.cos(p / (10000 ** ((2 * i)/d)))
        self.pe=Parameter(pe/(d**1/2*2),requires_grad=False)
        self.scale=Parameter(torch.ones(size=(1,d,1)))
    def forward(self,x):
        shape=x.shape
        x=self.pe[:,0:shape[1],0:shape[2]]*self.scale
        return x
class Learned_Encoding_Like(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        self.pe = Parameter(torch.zeros(1,hparams['affine_dim'],hparams['batch_size']))
    def forward(self,x):
        shape=x.shape
        return x+self.pe[:,:,0:shape[2]]
    def inverse(self,x):
        shape=x.shape
        return (x-self.pe[:,:,0:shape[2]],0)
class Multi_head_self_attention(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        s=hparams['affine_dim']
        self.keyconv=torch.nn.Conv1d(s,s, 1)
        self.queriesconv=torch.nn.Conv1d(s,s, 1)
        self.softmax=torch.nn.Softmax(dim=-1)
        self.projecta=torch.nn.Conv1d(s,s, 1)
    def split(self,x,shape):
        x=torch.reshape(x,(shape[0]*self.hparams['heads'],-1,shape[2]))
        return x
    def join(self,x,shape):
        x=torch.reshape(x,(shape[0],-1,shape[2]))
        return x
    def forward(self,x):
        shape=x.shape
        keys=self.keyconv(x)
        keys=self.split(keys,shape)
        queries=self.queriesconv(x)
        queries=self.split(queries,shape)
        values=torch.matmul(keys.permute(0,2,1),queries)
        seqlen=shape[2]
        attn=self.softmax(values/(seqlen**1/2))
        x=self.split(x,shape)
        out=torch.matmul(x,attn)
        out=self.join(out,shape)
        out=self.projecta(out)
        return out
class Multi_head_attention(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        s=hparams['affine_dim']
        self.keyconv=torch.nn.Conv1d(s,s, 1)
        self.queriesconv=torch.nn.Conv1d(s,s, 1)
        self.softmax=torch.nn.Softmax(dim=-1)
        self.projecta=torch.nn.Conv1d(s,s, 1)
    def split(self,x,shape):
        x=torch.reshape(x,(shape[0]*self.hparams['heads'],-1,shape[2]))
        return x
    def join(self,x,shape):
        x=torch.reshape(x,(shape[0],-1,shape[2]))
        return x
    def forward(self,x,y): #X will be sequence with attention over it, y will be the querying sequence
        shape=x.shape
        keys=self.keyconv(x)
        keys=self.split(keys,shape)
        queries=self.queriesconv(y)
        queries=self.split(queries,shape)
        values=torch.matmul(keys.permute(0,2,1),queries)
        seqlen=shape[2]
        attn=self.softmax(values/(seqlen**1/2))
        x=self.split(x,shape)
        out=torch.matmul(x,attn)
        out=self.join(out,shape)
        out=self.projecta(out)
        return out
class Attn_conv(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        self.act=torch.nn.ReLU()
        self.conva=torch.nn.Conv1d(hparams['affine_dim'],hparams['affine_dim'],1)
        self.convb=torch.nn.Conv1d(hparams['affine_dim'],hparams['affine_dim'],1)
    def forward(self,x):
        x=self.conva(x)
        x=self.act(x)
        x=self.convb(x)
        return x
class Attn_block(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        self.act=torch.nn.ReLU()
        self.attn=Multi_head_self_attention(hparams)
        self.conv=Attn_conv(hparams)
    def forward(self,x):
        add=x
        x=self.attn(x)
        x=x+add
        add=x
        x=self.conv(x)
        x=x+add
        return x

class Res_Unit(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        self.act=torch.nn.ReLU()
        self.conva=torch.nn.Conv1d(hparams['dim']//2,hparams['dim']//2,3,padding=1)
        self.convb=torch.nn.Conv1d(hparams['dim']//2,hparams['dim']//2,3,padding=1)
    def forward(self,x):
        add=x
        x=self.act(x)
        x=self.conva(x)
        x=self.act(x)
        x=self.convb(x)
        x=x+add
        return x
class Head(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        self.conva=torch.nn.Conv1d(hparams['dim']//2,hparams['affine_dim'],1)
        self.convb=torch.nn.Conv1d(hparams['affine_dim'],hparams['affine_dim'],1)
        self.gate=torch.nn.Conv1d(hparams['affine_dim'],hparams['affine_dim'],1)
        self.encoding=Learned_Encoding_Like(hparams)
        self.attn=Multi_head_attention(hparams)
        self.act=torch.nn.ReLU()
    def forward(self,x,enc):
        x=self.conva(x)
        x=x+self.encoding(x)
        x=self.act(x)
        x=self.convb(x)
        q=self.attn(enc,x)
        g=torch.tanh(self.gate(q+x))
        x=g*q+x
        return x
class Tail(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        self.act=torch.nn.ReLU()
        self.multiply_conv=torch.nn.Conv1d(hparams['affine_dim'],hparams['dim']//2,1)
        self.add_conv=torch.nn.Conv1d(hparams['affine_dim'],hparams['dim']//2,1)
        self.mscale=nn.Parameter(torch.zeros(size=(1,hparams['dim']//2,1)))
        self.ascale=nn.Parameter(torch.zeros(size=(1,hparams['dim']//2,1)))
    def forward(self,x):
        x=self.act(x)
        m=torch.tanh(self.multiply_conv(x))*self.mscale
        a=self.add_conv(x)#*self.ascale
        return (m,a)
class Transform_block(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.head=Head(hparams)
        self.bodya=Attn_block(hparams)
        self.bodyb=Attn_block(hparams)
        #self.bodyc=Attn_block(hparams)
        self.tail=Tail(hparams)
    def forward(self,x,enc):
        x=self.head(x,enc)
        x=self.bodya(x)
        x=self.bodyb(x)
        x=self.tail(x)
        return x
class Basic_conv_block(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        self.act=torch.nn.ReLU()
        self.conva=torch.nn.Conv1d(hparams['dim']//2,hparams['dim']//2,3,padding=1)
        self.resa=Res_Unit(hparams)
        self.resb=Res_Unit(hparams)
        self.multiply_conv=torch.nn.Conv1d(hparams['dim']//2,hparams['dim']//2,1)
        self.add_conv=torch.nn.Conv1d(hparams['dim']//2,hparams['dim']//2,1)
        self.mscale=nn.Parameter(torch.zeros(size=(1,hparams['dim']//2,1)))
        self.ascale=nn.Parameter(torch.zeros(size=(1,hparams['dim']//2,1)))
    def forward(self,x):
        x=self.conva(x)
        x=self.resa(x)
        x=self.resb(x)
        x=self.act(x)
        m=self.multiply_conv(x)*self.mscale
        a=self.add_conv(x)#*self.ascale
        return (m,a)
class Affine1d(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        self.block=Transform_block(hparams)
    def forward(self,data,enc):
        x=data[:,0:self.hparams['dim']//2,:]
        y=data[:,self.hparams['dim']//2:self.hparams['dim'],:]
        (m,a)=self.block(x,enc)
        out=torch.cat((x,y*torch.exp(m)+a),dim=1)
        return out
    def inverse(self,data,enc):
        x=data[:,0:self.hparams['dim']//2,:]
        (m,a)=self.block(x,enc)
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
class Add_one(nn.Module):
    def __init__(self,hparams):
        super().__init__()
    def forward(self,x):
        return x+1
    def inverse(self,x):
        return (x-1,torch.tensor(0))
class FC_block(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        self.lin=Conv1d(hparams)#Conv1d(hparams)
        self.act=Affine1d(hparams)#Parametric_Affine
    def forward(self,x,enc):
        return self.act(self.lin(x),enc)

    def inverse(self,x,encd):
        postconv=self.act.inverse(x,encd)
        start=self.lin.inverse(postconv[0])
        logdet=postconv[1]+start[1]
        return (start[0],logdet)
    def nothing(self):
        return
class Encoder(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        encoder=[]
        encoder.append(torch.nn.Conv1d(hparams['dim'],hparams['affine_dim'],1))
        encoder.append(Learned_Encoding_Like(hparams))
        for i in range(hparams['enc_blocks']):
            encoder.append(Attn_block(hparams))
        encoder.append(torch.nn.ReLU())
        encoder.append(torch.nn.Conv1d(hparams['affine_dim'],hparams['affine_dim'],1))
        self.encoder=nn.ModuleList(encoder)
    def forward(self,x):
        for block in self.encoder:
            x=block(x)
        return x
class Net(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        blocks=[]
        for i in range(hparams['blocks']):
            blocks.append(FC_block(hparams))
        blocks.append(Conv1d(hparams))
        self.blocks=nn.ModuleList(blocks)
        self.encoder=Encoder(hparams)
    def forward(self,x,cond):
        enc=self.encoder(cond)
        for block in self.blocks:
            if(type(block)==FC_block):
                x=block(x,enc)
            else:
                x=block(x)
        return x
    def inverse(self,x,cond):
        enc=self.encoder(cond)
        state=[x,0] #current inverse, log determinant
        for block in reversed(self.blocks):
            inv=0
            if(type(block)==FC_block): #Only feed conditioning to the affine layers, not the conv1d or any other layer types
                inv=block.inverse(state[0],enc)
            else:
                inv=block.inverse(state[0])
            state[0]=inv[0]
            state[1]=state[1]+inv[1]
        return state
