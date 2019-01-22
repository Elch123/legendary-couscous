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
engbpe=BpEmbed(hparams,bpemb_en)
maker=Batch_maker("traindeen.pickle")
net=Net(hparams)
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
def make_batch(batch_size):
    batch=maker.make_batch(batch_size)[1]
    embedded=engbpe(torch.tensor(batch).long()).permute(0,2,1)
    print(embedded.shape)
    return embedded
def modelprint():
    print(make_batch(hparams['batch_size']))
    print(net(make_normal_batch(hparams['channels'],hparams['batch_size'])))
def verify():
    batch=make_batch(hparams['batch_size'])
    passed=net(net.inverse(batch)[0])
    passedtwo=net.inverse(net(batch))[0]
    print(torch.mean(batch-passed))
    print(torch.mean(batch-passedtwo))
#make_batch(1000)
def train():
    for e in range(hparams['batches']):
        #10*make_normal_batch(hparams['batch_size'])
        target=make_batch(hparams['batch_size'])
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
