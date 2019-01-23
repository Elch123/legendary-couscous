from loader import makeendeprocessors
(bpemb_en,bpemb_de)=makeendeprocessors()
import torch
import torch.nn as nn
import torch.nn.functional as F
from hparams import hparams
import numpy as np
from embed import BpEmbed
from makebatches import Batch_maker
from blocks import Net
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
engbpe=BpEmbed(hparams,bpemb_en)
maker=Batch_maker("traindeen.pickle")
net=Net(hparams)
net=net.to(device)
"""
print(net)
for p in net.parameters():
    print(p.shape)"""
optimizer = torch.optim.SGD(net.parameters(), lr=hparams['lr'], momentum=0.9,nesterov=True)
def make_normal_batch(batch_size,channels,seqlen):
    samples=channels*seqlen
    m = torch.distributions.MultivariateNormal(torch.zeros(samples), scale_tril=torch.eye(samples)) #zero mean, identity covariancm.samplee
    data=m.sample((batch_size,))
    data=data.reshape(batch_size,channels,seqlen)
    return data
def negative_log_gaussian_density(data):
    #this assumes a mean of zero, and a standard devation of one. The coefficient will probably be important, so I'm keeping that.
    size=data.shape[1]
    m = torch.distributions.MultivariateNormal(torch.zeros(size), scale_tril=torch.eye(size))
    #m = torch.distributions.MultivariateNormal(torch.zeros(size).to(device), scale_tril=torch.eye(size).to(device))
    nll=-m.log_prob(data.cpu())
    return nll
def make_batch(batch_size):
    batch=maker.make_batch(batch_size)[0] #English, not German
    embedded=engbpe(torch.tensor(batch).long()).permute(0,2,1)
    print(embedded.shape)
    return embedded
def decode_print(data):
    #data=data.permute(0,2,1)
    text=engbpe.disembed_batch(data.cpu())
    print(text)
def modelprint():
    b=make_batch(hparams['batch_size'])
    decode_print(b)
    decode_print(net(make_normal_batch(b.shape[0],b.shape[1],b.shape[2]).to(device)).cpu())
def print_numpy(description,data):
    print(description+str(data.detach().cpu().numpy()))
def verify():
    batch=make_batch(hparams['batch_size']).to(device)
    passed=net(net.inverse(batch)[0])
    passedtwo=net.inverse(net(batch))[0]
    print_numpy("Inverse first error ",torch.mean(batch-passed))
    print_numpy("Forward first error ",torch.mean(batch-passedtwo))
def flatten(x):
    return x.reshape(x.shape[0],-1)
#make_batch(1000)
def train():
    for e in range(hparams['batches']):
        #10*make_normal_batch(hparams['batch_size'])
        target=make_batch(hparams['batch_size'])
        target=target.to(device)
        #target.requires_grad=True
        #print(target)
        start=net.inverse(target)
        distloss=negative_log_gaussian_density(flatten(start[0]))/hparams['batch_size']
        print(distloss)
        #print(distloss.shape)
        jacloss=torch.sum(start[1],dim=1)/hparams['batch_size'].cpu()
        print(jacloss)
        #print(jacloss.shape)
        loss=distloss+jacloss
        loss=torch.mean(loss)
        loss*=torch.tensor(2)
        net.zero_grad()
        print(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()
        if(e%1==0):
            print_numpy("distribution loss " ,torch.mean(distloss))
            print_numpy("Transformation loss " ,torch.mean(jacloss))
            print_numpy("Total loss ",loss)
        if(e%10==0):
            verify()
            modelprint()


train()


torch.cuda.synchronize()
del net
torch.cuda.synchronize()
torch.cuda.empty_cache()
torch.cuda.synchronize()
