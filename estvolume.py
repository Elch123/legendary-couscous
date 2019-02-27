import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from esthparams import hparams
import matplotlib.pyplot as plt
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
        self.m_scale=nn.Parameter(torch.zeros(size=(hparams['channels']//2,)))
        #self.add_scale=nn.Parameter(torch.zeros(size=(hparams['channels']//2,)))
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
        m=self.mults(x)*self.m_scale
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
#print(x)
#print(net(x))
def make_batch(size,batch_size):
    dist=torch.distributions.OneHotCategorical(probs=torch.ones(size=(size,)))
    m = torch.distributions.MultivariateNormal(torch.zeros(size), scale_tril=torch.eye(size))
    target=1*dist.sample((batch_size,))+.1*m.sample((batch_size,))
    return target
def modelprint():
    with torch.no_grad():
        print(make_batch(hparams['channels'],hparams['batch_size']))
        print(net(make_normal_batch(hparams['channels'],hparams['batch_size'])))
def graph_out(points,bound):
    in_array=np.zeros(shape=(points,points,2))
    with torch.no_grad():
        for i in range(points):
            for j in range(points):
                in_array[i][j][0]=(i-points/2)/points*bound*2
                in_array[i][j][1]=(j-points/2)/points*bound*2
        print(in_array)
        in_array=in_array.reshape((points**2,2))
        in_array=torch.tensor(in_array).float()
        out=net(in_array).detach().numpy()
        out=out.reshape((points,points,2))
        plt.imshow(out[:,:,0])
        plt.show()
        plt.imshow(out[:,:,1])
        plt.show()
        print(out)
def trace_out(start,direction,delta,num_points):
    in_array=np.zeros(shape=(num_points,2))
    direction=direction/np.sqrt(np.sum(direction**2))
    with torch.no_grad():
        for i in range(num_points):
            in_array[i]=start+direction*delta*i
        print(in_array)
        in_array=torch.tensor(in_array).float()
        out=net(in_array).detach().numpy()
        plt.plot(out)
        plt.show()
def mismatch(outa,outb):
    ga=outa[0]>outa[1]
    gb=outb[0] > outb[1]
    if(ga != gb):
        return True
    return False
def find_root(fn,start,direction,start_scale=torch.tensor(1e-6)):
    #in_array=np.zeros(shape=(num_points,2))
    with torch.no_grad():
        out=fn(start).detach().numpy()
        initial_out=out
        while(start_scale<1e4):
            candidate_point=start+direction*start_scale
            out=fn(candidate_point).detach().numpy()
            if(mismatch(initial_out,out)):
                return start_scale,candidate_point
            start_scale*=2
def distance(vector):
    return torch.sum(vector**2)**(1/2)
def neg_log_density(point):
    dimentions=len(point)
    log_gaussian_density=-torch.reduce_sum(point**2)-1/2*torch.log(torch.tensor(2*3.14159))
    #volume=dimentions*torch.log(dist)
    density=-log_gaussian_density
    return density
def neg_log_p(point,dist):
    dimentions=len(point)
    log_gaussian_density=-torch.sum(point**2)-1/2*torch.log(torch.tensor(2*3.14159))
    volume=(dimentions-1)*torch.log(dist)
    density=log_gaussian_density+volume
    return density
def basic_integrate(point,direction,dist,samples=1000):
    logarithms=[]
    dimentions=len(point)
    for i in range(samples):
        cur_d=dist/samples
        logarithms.append(neg_log_p(point+direction*cur_d,cur_d)-torch.log(samples)) #-torch.log(samples) to divide by the number of samples
    logarithm=logsumexp(logarithms)
    return logarithm
def argmax_logs(logs):
    l=0
    argmax=logs[0][2]
    for i in range(len(logs)):
        if(logs[i][2]>argmax):
            argmax=logs[i][2]
            l=i
    print(l)
    return l
def logsumexp(tensor): #implements logsumexp trick for numerically stable adding of logarithms
    maximum=torch.max(tensor)
    tensor-=maximum
    remaider_log_sum=torch.log(torch.sum(torch.exp(tensor)))
    result=remaider_log_sum+maximum
    return result
def adaptive_integrate(point,direction,dist,samples=1000):
    log_two=torch.log(torch.tensor(2.0))
    logarithms=[]
    logarithms.append([dist,0,neg_log_p(point+direction*dist,dist)])
    for i in range(samples):
        l=argmax_logs(logarithms)
        logarithms[l][1]+=1
        logarithms[l][2]-=log_two
        new_dist=logarithms[l][0]-dist*(2**-logarithms[l][1])
        logarithms.append([new_dist,logarithms[l][1],neg_log_p(point+direction*new_dist,new_dist)-log_two*logarithms[l][1]])
    chunks=[]
    for i in range(len(logarithms)):
        chunks.append(logarithms[i][2])
    print(chunks)
    chunks=torch.tensor(chunks)
    integral=logsumexp(chunks)
    return integral
def test_fn(point):
    d=distance(point)-1/2
    s=torch.sigmoid(d)
    return torch.tensor([s,1-s])
def random_dir_vector(size):
    x=torch.randn(size)
    x=x/distance(x)
    return x
def make_grad(fn,center):
    direction=random_dir_vector(['size'])
    root=find_root(fn,center,direction) #distance, and point of intersection
    prob_integral=adaptive_integrate(center,direction,root[0])
    end_p=neg_log_p(root[1],root[0])
    grad=end_p/prob_integral*direction
    return (root[1],grad)
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
        if(loss<50):
            loss.backward()
        #torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()

#verify()
#train()
x=torch.tensor([3.0,4.0])
#print(distance(x))
x=random_dir_vector(2)*torch.randn(1)

#print(x)
#print(test_fn(x))
init_p=torch.tensor([0.0,0.0])
"""for i in range(10000):
    j=torch.tensor(i/100)
    print_p=torch.tensor([0.0,j])
    print(j)
    print(neg_log_p(print_p,j))
    print()"""
test_direction=random_dir_vector(2)
d=find_root(test_fn,init_p,test_direction)
log_vol_estimate=adaptive_integrate(init_p,test_direction,d[0]*.2,10000)
print(log_vol_estimate)
#verify()
#graph_out(10,1)
#trace_out(np.array((0,0)),np.array((1,1)),.1,20)
#print(net.lin.linweight)
