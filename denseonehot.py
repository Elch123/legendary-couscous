import torch
import torch.nn as nn
import torch.nn.functional as F
#print(dir(F))
hparams={
'channels':10,
'lr':.001,
'batches':500,
'batch_size':10,
'blocks':5
}
#For all these classes, the inverse is the first return, the log determinant is the second
 #HAHA! make sure to be smart with the means I am taking, and not take a mean over the batch dimesion!!!
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
        self.alphas=nn.Parameter(torch.ones(size=(hparams['channels'],)))
    def forward(self,x):
        return F.prelu(x,torch.exp(self.alphas))
    def inverse(self,x):
        inv=F.prelu(x,1/torch.exp(self.alphas))
        logdet=(inv>=0).float()*torch.ones(size=(self.hparams['channels'],))+(inv<0).float()*self.alphas
        logdet=torch.mean(logdet,dim=1)
        #print("Prelu logdet "+str(logdet))
        return (inv,logdet)
class FC_block(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        self.lin=Lin_bidirectional(hparams)
        self.act=Prelu(hparams)
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
        blocks=[FC_block(hparams) for i in range(hparams['blocks'])]
        print(blocks)
        self.blocks=nn.ModuleList(blocks)
    def forward(self,x):
        for block in self.blocks:
            x=block(x)
        return x
    def inverse(self,x):
        state=[x,torch.tensor(0.0,requires_grad=True)] #current inverse, log determinant
        for i in range(self.hparams['blocks']-1,0,-1):
            inv=self.blocks[i].inverse(state[0])
            #print(inv[1])
            state[0]=inv[0]
            state[1]=state[1]+inv[1]
        return state
def make_normal_batch(size):
    m = torch.distributions.MultivariateNormal(torch.zeros(size), scale_tril=torch.eye(size)) #zero mean, identity covariancm.samplee
    data=m.sample((100,))
    return data
def avg_negative_log_gaussian_density(data):
    #this assumes a mean of zero, and a standard devation of one. The coefficient will probably be important, so I'm keeping that.
    size=data.shape[1]
    m = torch.distributions.MultivariateNormal(torch.zeros(size), scale_tril=torch.eye(size)) #One over sqrt of 2 pi
    nll=-m.log_prob(data)/size
    return nll
net=FC_net(hparams)
optimizer = torch.optim.SGD(net.parameters(), lr=hparams['lr'], momentum=0.9,nesterov=True)
x=make_normal_batch(10)
def make_batch(size):
    dist=torch.distributions.OneHotCategorical(probs=torch.ones(size=(size,)))
    m = torch.distributions.MultivariateNormal(torch.zeros(size), scale_tril=torch.eye(size))
    target=1*dist.sample((100,))+.01*m.sample((100,))
    return target
for e in range(hparams['batches']):
    #10*make_normal_batch(hparams['batch_size'])
    target=make_batch(hparams['channels'])
    #print(target)
    start=net.inverse(target)
    distloss=avg_negative_log_gaussian_density(start[0])
    #print("distribution loss " + str(distloss))
    loss=distloss+start[1]
    loss=torch.mean(loss)
    print(loss)
    net.zero_grad()
    loss=loss
    print(loss)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
    optimizer.step()
print(net(make_normal_batch(hparams['batch_size'])))
#print(net.lin.linweight)
