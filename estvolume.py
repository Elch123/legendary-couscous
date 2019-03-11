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
    #print(outa.shape)
    init=outa[:,0]>outa[:,1]
    now=outb[:,0]>outb[:,1]
    changed=(init!=now)
    #print(changed)
    return changed
def find_root_batch(fn,start,direction,start_scale=1e-3):
    #in_array=np.zeros(shape=(num_points,2))
    scales=torch.zeros(size=(hparams['batch_size'],1))
    uppers=torch.zeros_like(scales)
    lowers=torch.zeros_like(scales)
    never=torch.zeros(size=(hparams['batch_size'],))
    never[:]=1
    scales[:]=start_scale
    #print("start scale" + str(start_scale))
    with torch.no_grad():
        out=fn(start).detach()
        initial_out=out
        #print(initial_out)
        for i in range(30):
            #print(start.shape)
            #print(direction.shape)
            #print(scales.shape)
            dists=direction*scales
            candidate_point=start+dists
            out=fn(candidate_point)
            #print(candidate_point.shape)
            #print(out)
            match=mismatch(initial_out,out)
            for (j,val) in enumerate(match):
                if(never[j]):
                    if(not val):
                        scales[j]*=2
                    if(val):
                        uppers[j]=scales[j]
                        lowers[j]=scales[j]/2
                        never[j]=0
                else:
                    if(val):
                        uppers[j]=scales[j]
                    else:
                        lowers[j]=scales[j]
                    scales[j]=(uppers[j]+lowers[j])/2
            for j in range(len(scales)):
                if(scales[j]>50):
                    scales[j]=50 #clip scales array so the integration routine will be accurate
        dists=direction*scales
        candidate_point=start+dists
        return scales,candidate_point
def neg_log_density(point):
    dimentions=len(point)
    log_gaussian_density=-torch.reduce_sum(point**2)-1/2*torch.log(torch.tensor(2*3.14159))
    #volume=dimentions*torch.log(dist)
    density=-log_gaussian_density
    return density

def neg_log_p_batch(point,dist):
    dimentions=point.shape[-1]
    log_gaussian_density=-torch.sum(point**2,dim=-1)/2-1/2*torch.log(torch.tensor(2*3.14159))
    volume=(dimentions-1)*torch.log(dist)
    density=log_gaussian_density+volume
    return density
def logsumexp(tensor): #implements logsumexp trick for numerically stable adding of logarithms
    maximum=torch.max(tensor)
    tensor-=maximum
    remaider_log_sum=torch.log(torch.sum(torch.exp(tensor)))
    result=remaider_log_sum+maximum
    return result
def logsumexp_batch(tensor): #implements logsumexp trick for numerically stable adding of logarithms
    maximum=torch.max(tensor,dim=-1)[0]

    tensor-=torch.unsqueeze(maximum,dim=-1)
    remaider_log_sum=torch.log(torch.sum(torch.exp(tensor),dim=-1))
    result=remaider_log_sum+maximum
    return result
def fast_basic_integrate_batch(point,direction,dist,samples=1000):
    point=torch.unsqueeze(point,dim=-2)
    direction=torch.unsqueeze(direction,dim=-2)
    eval_dists=torch.stack([torch.linspace(0,d.item(),steps=samples) for d in dist],dim=0)
    eval_dists_expanded=torch.unsqueeze(eval_dists,dim=-1)
    eval_points=point+direction*eval_dists_expanded
    results=neg_log_p_batch(eval_points,eval_dists)
    #print(results.shape)
    #print("dist is "+str(dist))
    eval_sums=logsumexp_batch(results)
    #print("eval sums shape is "+str(eval_sums.shape))
    #print("eval sums is "+str(eval_sums))
    distlog=torch.log(torch.squeeze(dist))
    integral=eval_sums-torch.log(torch.tensor(samples).float())+distlog
    #print("integral shape is "+str(integral.shape))
    return integral
def distance(vector):
    return torch.sum(vector**2,dim=-1)**(1/2)
def test_fn(batch):
    comp=torch.zeros_like(batch)
    comp[:]=batch[:]
    comp[:,1]*=2
    d=distance(comp)
    return torch.stack([d,1-d],dim=-1)
def random_dir_vector_batch(batch_size,size):
    x=torch.stack([torch.randn(size) for i in range(batch_size)],dim=0)
    x=x/torch.unsqueeze(distance(x),dim=-1)
    return x
def make_grad_batch(fn,center):
    delta=.01 #how for back to go for finite differences calculation
    direction=random_dir_vector_batch(hparams['batch_size'],hparams['size'])
    #print("random direction shape is " + str(direction.shape))
    root=find_root_batch(fn,center,direction) #distance, and point of intersection
    print("root  is " + str(root))
    prob_integral=fast_basic_integrate_batch(center,direction,root[0])
    prob_integral_two=fast_basic_integrate_batch(center-direction*delta,direction,root[0]+delta) #compute gradient of first point using finite differences
    #print(root[1].shape)
    #print(root[0].shape)
    end_p=neg_log_p_batch(root[1],torch.squeeze(root[0]))
    print("end p is " + str(end_p))
    print("logarithm of prob integral one is " + str(prob_integral))
    print("logarithm of prob integral two is " + str(prob_integral_two))
    grad_end_mag=torch.exp(end_p-prob_integral)
    grad_end_mag=torch.unsqueeze(grad_end_mag,dim=-1)
    print("gradient magnitude at end is " + str(grad_end_mag))
    grad_end=grad_end_mag*direction #Use formula for graient at end
    #grad_start=(torch.exp(torch.tensor(prob_integral_two))-torch.exp(torch.tensor(prob_integral)))/delta*-direction
    grad_start_mag=torch.tensor(prob_integral_two-prob_integral)/delta
    grad_start_mag=torch.unsqueeze(grad_start_mag,dim=-1)
    print("gradient magnitude at start is " + str(grad_start_mag))
    grad_start=-grad_start_mag*direction #derivative of negative log of likelihood w/ finite differences. Negative, because in opposite direction of random vector
    if(torch.sum(grad_start**2)>500):
        grad_start=torch.zeros_like(grad_start)
    return (root[1],grad_end,center,grad_start)
def verify():
    batch=make_batch(hparams['channels'],hparams['batch_size'])
    passed=net(net.inverse(batch)[0])
    passedtwo=net.inverse(net(batch))[0]
    print(torch.mean(batch-passed))
    print(torch.mean(batch-passedtwo))
init_p=torch.zeros(size=(hparams['batch_size'],2))
def train():
    for e in range(hparams['batches']):
        #10*make_normal_batch(hparams['batch_size'])
        target=make_batch(hparams['channels'],hparams['batch_size'])
        #print(target)
        with torch.no_grad():
            start=net.inverse(target)[0]
        #grads=make_grad_batch(net,start)
        grads=make_grad_batch(test_fn,init_p)
        print(grads)
        if(e%500==0):
            #verify()
            #modelprint()
            pass
        net.zero_grad()
        if(loss<50):
            loss.backward()
        #torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()

#verify()
#train()

#print(x)
#print(test_fn(x))
init_p=torch.zeros(size=(hparams['batch_size'],2))
test_direction=random_dir_vector_batch(hparams['batch_size'],2)
#print(distance(test_direction))
#d=find_root_batch(test_fn,init_p,test_direction)
#print(d)
#d=find_root(test_fn,init_p,test_direction)
#print(make_grad(test_fn,init_p))
#graph_out(100,10)
#train_online()
train()
#verify()
