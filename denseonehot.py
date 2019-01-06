import torch
import torch.nn as nn
import torch.nn.functional as F
#print(dir(F))
hparams={
'channels':10,
'lr':.005,
'batches':10000,
'batch_size':10,
'blocks':3
}
#For all these classes, the inverse is the first return, the log determinant is the second
 #HAHA! make sure to be smart with the means I am taking, and not take a mean over the batch dimesion!!!
 #Sum logdet over channel dimension, don't mean over batch
class Lin_bidirectional(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        self.linweight=nn.Parameter(torch.empty(size=(hparams['channels'],hparams['channels'])))
        torch.nn.init.orthogonal_(self.linweight)
        self.register_parameter("w",self.linweight)
    def forward(self,x):
        return F.linear(x,self.linweight)
    def inverse(self,x):
        invlin=F.linear(x,torch.inverse(self.linweight))
        logdet=torch.slogdet(self.linweight)[1]#/self.hparams['channels']
        #print("Conv logdet " + str(logdet))
        return (invlin,logdet)
class Prelu(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        self.alphas=nn.Parameter(torch.zeros(size=(hparams['channels'],)))
    def forward(self,x):
        y=F.prelu(x,torch.exp(self.alphas))
        return y
    def inverse(self,x):
        inv=F.prelu(x,1/torch.exp(self.alphas))
        logdet=(inv>=0).float()*torch.ones(size=(self.hparams['channels'],))+(inv<0).float()*self.alphas
        logdet=torch.sum(logdet,dim=1)
        #print("Prelu logdet "+str(logdet))
        return (inv,logdet)
class Square(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
    def forward(self,x):
        y=x**2*torch.sign(x).float()
        #print(y)
        return y
    def inverse(self,x):
        #print(x)
        inv=torch.abs(x)**(1/2)*torch.sign(x).float()
        logdet=torch.sum(torch.log(2*torch.abs(inv)),dim=1)
        #print(torch.mean(logdet))
        return (inv,logdet)
class Addone(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
    def forward(self,x):
        y=x+1
        #print(y)
        return y
    def inverse(self,x):
        #print(x)
        inv=x-1
        logdet=0
        #print(torch.mean(logdet))
        return (inv,logdet)

class FC_block(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        self.lin=Lin_bidirectional(hparams)
        self.act=Square(hparams)
    def forward(self,x):
        return self.act(self.lin(x))
    def inverse(self,x):
        postconv=self.act.inverse(x)
        start=self.lin.inverse(postconv[0])
        logdet=postconv[1]+start[1]
        return (start[0],logdet)
class FC_net(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        blocks=[FC_block(hparams) for i in range(hparams['blocks'])] #FC_block
        blocks.append(Lin_bidirectional(hparams))
        print(blocks)
        self.blocks=nn.ModuleList(blocks)
    def forward(self,x):
        for block in self.blocks:
            x=block(x)
        return x
    def inverse(self,x):
        state=[x,torch.tensor(0.0,requires_grad=True)] #current inverse, log determinant
        for i in range(self.hparams['blocks']-1,-1,-1):
            inv=self.blocks[i].inverse(state[0])
            #print(inv[1])
            state[0]=inv[0]
            state[1]=state[1]+inv[1]
        return state
def make_normal_batch(size):
    m = torch.distributions.MultivariateNormal(torch.zeros(size), scale_tril=torch.eye(size)) #zero mean, identity covariancm.samplee
    data=m.sample((100,))
    return data
def negative_log_gaussian_density(data):
    #this assumes a mean of zero, and a standard devation of one. The coefficient will probably be important, so I'm keeping that.
    size=data.shape[1]
    m = torch.distributions.MultivariateNormal(torch.zeros(size), scale_tril=torch.eye(size)) #One over sqrt of 2 pi
    nll=-m.log_prob(data)
    return nll
net=FC_net(hparams)
optimizer = torch.optim.SGD(net.parameters(), lr=hparams['lr'], momentum=0.9,nesterov=True)
x=make_normal_batch(10)
def make_batch(size):
    dist=torch.distributions.OneHotCategorical(probs=torch.ones(size=(size,)))
    m = torch.distributions.MultivariateNormal(torch.zeros(size), scale_tril=torch.eye(size))
    target=0*dist.sample((100,))+torch.abs(1.0*m.sample((100,)))
    return target
def train():
    for e in range(hparams['batches']):
        #10*make_normal_batch(hparams['batch_size'])
        target=make_batch(hparams['channels'])
        #print(target)
        start=net.inverse(target)
        distloss=negative_log_gaussian_density(start[0])/hparams['channels']
        print("distribution loss " + str(torch.mean(distloss)))
        jacloss=start[1]/hparams['channels']
        print("Transformation loss " + str(torch.mean(jacloss)))
        loss=distloss+jacloss
        loss=torch.mean(loss)
        print(loss)
        net.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()
def verify():
    batch=make_batch(hparams['batch_size'])
    passed=net(net.inverse(batch)[0])
    passedtwo=net.inverse(net(batch))[0]
    print(torch.mean(batch-passed))
    print(torch.mean(batch-passedtwo))
verify()
#train()
#print(net(make_normal_batch(hparams['batch_size'])))

#print(net.lin.linweight)
