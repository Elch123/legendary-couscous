import torch
import torch.nn as nn
import torch.nn.functional as F
#print(dir(F))
hparams={
'channels':10,
'lr':.0003,
'batches':500001,
'batch_size':50,
'blocks':60
}
torch.set_printoptions(threshold=100000)
#For all these classes, the inverse is the first return, the log determinant is the second
 #HAHA! make sure to be smart with the means I am taking, and not take a mean over the batch dimesion!!!
 #Sum logdet over channel dimension, don't mean over batch
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
class Bent(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        self.alphas=nn.Parameter(torch.ones(size=(hparams['channels'],)))
        self.betas=nn.Parameter(torch.ones(size=(hparams['channels'],)))
        self.bias=nn.Parameter(torch.zeros(size=(hparams['channels'],)))
    def forward(self,x):
        x=x+self.bias
        y=x*(x>=0).float()*self.alphas+x*(x<0).float()*self.betas
        return y
    def inverse(self,x):
        inv=x
        inv=inv*(inv>=0).float()/self.alphas+inv*(inv<0).float()/self.betas
        inv=inv-self.bias
        logdet=(inv>=0).float()*torch.log(self.alphas)+(inv<0).float()*torch.log(self.betas)
        logdet=torch.sum(logdet,dim=1)
        #print("Prelu logdet "+str(logdet))
        return (inv,logdet)
class Soft_bent(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        self.alphas=nn.Parameter(torch.zeros(size=(hparams['channels'],)))
        self.betas=nn.Parameter(torch.zeros(size=(hparams['channels'],)))
        self.bias=nn.Parameter(torch.zeros(size=(hparams['channels'],)))
    def forward(self,x):
        y=torch.exp(self.alphas)*torch.log1p(torch.exp(x))-torch.exp(self.betas)*torch.log1p(torch.exp(-x))
        return y
    def derivative(self,x):
        d=(torch.exp(self.alphas)*torch.exp(x)+torch.exp(self.betas))/(1+torch.exp(x))
        return d
    def error(self,x,target):
        return(self.forward(x)-target)
    def inverse(self,y):
        target=y
        x=y
        for i in range(3):
            x=x-self.error(x,target)/self.derivative(x)
            #print(x)
        logdet=torch.sum(torch.log(self.derivative(x)),dim=1)
    #    print("Prelu derivative "+str(logdet))
        return (x,logdet)
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
class Parametric_Affine(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        self.act=torch.nn.ReLU()
        self.lina=torch.nn.Linear(hparams['channels']//2,hparams['channels'])
        self.linb=torch.nn.Linear(hparams['channels'],hparams['channels'])
        self.posmults=torch.nn.Linear(hparams['channels'],hparams['channels']//2)
        self.negmults=torch.nn.Linear(hparams['channels'],hparams['channels']//2)
        self.adds=torch.nn.Linear(hparams['channels'],hparams['channels']//2)
        self.posgate=nn.Parameter(torch.zeros(size=(hparams['channels']//2,)))
        self.neggate=nn.Parameter(torch.zeros(size=(hparams['channels']//2,)))
    def compute_nets(self,x):
        x=self.act(x)
        x=self.lina(x)
        x=self.act(x)
        posm=self.posmults(x)*self.posgate
        negm=self.negmults(x)*self.neggate
        a=self.adds(x)
        return (posm,negm,a)
    def makemult(self,pos,neg,y):
        m=(y>=0).float()*pos+(y<0).float()*neg
        return m
    def forward(self,data):
        x=data[:,0:self.hparams['channels']//2]
        y=data[:,self.hparams['channels']//2:self.hparams['channels']]
        (posm,negm,a)=self.compute_nets(x)
        m=self.makemult(posm,negm,y)
        newy=y*torch.exp(m)+a
        data=torch.cat((x,newy),dim=1)#
        return data
    def inverse(self,data):
        x=data[:,0:self.hparams['channels']//2]
        y=data[:,self.hparams['channels']//2:self.hparams['channels']]
        (posm,negm,a)=self.compute_nets(x)
        y=y-a
        m=self.makemult(posm,negm,y)
        data=torch.cat((x,y/torch.exp(m)),dim=1)
        logdet=torch.sum(m,dim=1)
        return (data,logdet)
class Square(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
    def forward(self,x):
        y=x**2*torch.sign(x).float()/2
        #print(y)
        return y
    def inverse(self,x):
        #print(x)
        x*=2
        inv=torch.abs(x)**(1/2)*torch.sign(x).float()
        logdet=torch.sum(torch.log(torch.abs(inv)),dim=1)
        #print(torch.mean(logdet))
        return (inv,logdet)
class Tan(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
    def forward(self,x):
        y=torch.tan(x)
        return y
    def inverse(self,x):
        inv=torch.atan(x)
        logdet=torch.sum(-torch.log(torch.cos(inv)**2),dim=1)
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
class Identity(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
    def forward(self,x):
        return x
    def inverse(self,x):
        #print(x)
        inv=x
        logdet=0
        #print(torch.mean(logdet))
        return (inv,logdet)
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
        for block in reversed(self.blocks):
            inv=block.inverse(state[0])
            state[0]=inv[0]
            state[1]=state[1]+inv[1]
        return state

net=FC_net(hparams)
#net=Soft_bent(hparams)
optimizer = torch.optim.SGD(net.parameters(), lr=hparams['lr'], momentum=0.9,nesterov=True)
def make_normal_batch(size,batch_size):
    m = torch.distributions.MultivariateNormal(torch.zeros(size), scale_tril=torch.eye(size)) #zero mean, identity covariancm.samplee
    data=m.sample((batch_size,))
    return data
def negative_log_gaussian_density(data):
    #this assumes a mean of zero, and a standard devation of one. The coefficient will probably be important, so I'm keeping that.
    size=data.shape[1]
    m = torch.distributions.MultivariateNormal(torch.zeros(size), scale_tril=torch.eye(size)) #One over sqrt of 2 pi
    nll=-m.log_prob(data)
    return nll
x=make_normal_batch(hparams['channels'],hparams['batch_size'])
print(x)
print(net(x))
def make_batch(size,batch_size):
    dist=torch.distributions.OneHotCategorical(probs=torch.ones(size=(size,)))
    m = torch.distributions.MultivariateNormal(torch.zeros(size), scale_tril=torch.eye(size))
    target=1*dist.sample((batch_size,))+.1*m.sample((batch_size,))
    return target
def modelprint():
    print(make_batch(hparams['channels'],hparams['batch_size']))
    print(net(make_normal_batch(hparams['channels'],hparams['batch_size'])))
def verify():
    batch=make_batch(hparams['channels'],hparams['batch_size'])
    passed=net(net.inverse(batch)[0])
    passedtwo=net.inverse(net(batch))[0]
    print(torch.mean(batch-passed))
    print(torch.mean(batch-passedtwo))
def train():
    for e in range(hparams['batches']):
        #10*make_normal_batch(hparams['batch_size'])
        target=make_batch(hparams['channels'],hparams['batch_size'])
        #print(target)
        start=net.inverse(target)
        distloss=negative_log_gaussian_density(start[0])/hparams['channels']
        jacloss=start[1]/hparams['channels']
        loss=distloss+jacloss
        loss=torch.mean(loss)
        if(e%1==0):
            print("distribution loss " + str(torch.mean(distloss)))
            print("Transformation loss " + str(torch.mean(jacloss)))
            print(loss)
        if(e%500==0):
            verify()
            modelprint()
        net.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()

verify()
train()
verify()


#print(net.lin.linweight)
