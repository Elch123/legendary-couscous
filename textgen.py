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
import tracemalloc
import time
#tracemalloc.start()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print(device)
engbpe=BpEmbed(hparams,bpemb_en)
debpe=BpEmbed(hparams,bpemb_de)
maker=Batch_maker("traindeen.pickle")
net=Net(hparams)
net=net.to(device)
"""
Updating the net for translation
figure out how to wire the parts together
translate!
"""
#optimizer = torch.optim.SGD(net.parameters(), lr=hparams['lr'], momentum=0.9,nesterov=True)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
def make_normal_batch(batch_size,channels,seqlen):
    samples=seqlen
    m = torch.distributions.MultivariateNormal(torch.zeros(samples), scale_tril=torch.eye(samples)) #zero mean, identity covariancm.samplee
    data=m.sample((batch_size,channels))
    #data=data.reshape(batch_size,channels,seqlen)
    return data
def make_normal_batch_like(b):
    return make_normal_batch(b.shape[0],b.shape[1],b.shape[2])
def negative_log_gaussian_density(data):
    #this assumes a mean of zero, and a standard devation of one. The coefficient will probably be important, so I'm keeping that.
    size=data.shape[-1]
    #m = torch.distributions.MultivariateNormal(torch.zeros(size), scale_tril=torch.eye(size))
    m = torch.distributions.MultivariateNormal(torch.zeros(size).to(device), scale_tril=torch.eye(size).to(device))
    nll=-m.log_prob(data)
    return nll
def make_batch(batch_size):
    batch=maker.make_batch(batch_size) #English=0 not German=1
    target=engbpe(torch.tensor(batch[0]).long()).permute(0,2,1)
    source=debpe(torch.tensor(batch[1]).long()).permute(0,2,1)
    noise=make_normal_batch_like(target)*hparams['noise_scale']
    target+=noise
    #embedded_batch=torch.stack((target,source),dim=0)
    print("batch shape " + str(target.shape))
    return (target.to(device),source.to(device))
def decode_print(data):
    #data=data.permute(0,2,1)
    text=engbpe.disembed_batch(data.cpu())
    print(text)
def modelprint():
    b=make_batch(hparams['batch_size'])
    decode_print(b[0])
    with torch.no_grad():
        decode_print(net(make_normal_batch_like(b[0]).to(device),b[1]))
def print_numpy(description,data):
    print(description+str(data.detach().cpu().numpy()))
def verify():
    batch=make_batch(hparams['batch_size'])
    with torch.no_grad():
        passed=net(net.inverse(batch[0],batch[1])[0],batch[1])
        print("")
        passedtwo=net.inverse(net(batch[0],batch[1]),batch[1])[0]
        erra=torch.mean(batch[0]-passed)
        errb=torch.mean(batch[0]-passedtwo)
    print_numpy("Inverse first error ",erra)
    print_numpy("Forward first error ",errb)
def print_alloced():
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    print("[ Top 100 ]")
    for stat in top_stats[:100]:
        print(stat)
def prof_forward():
    batch=make_batch(hparams['batch_size'])
    with torch.no_grad():
        with torch.autograd.profiler.profile() as prof:
        #with torch.autograd.profiler.emit_nvtx() as prof:
            passed=net(net.inverse(batch[0],batch[1])[0],batch[1])
        print(prof)
def train():
    for e in range(hparams['batches']):
        if(e%30==0):
            #prof_forward()
            #verify()
            modelprint()
        target=make_batch(hparams['batch_size'])
        start=net.inverse(target[0],target[1])
        distloss=negative_log_gaussian_density(start[0].permute(0,2,1))/hparams['dim']
        jacloss=start[1]/hparams['dim']
        loss=distloss+jacloss
        loss=torch.mean(loss)
        #print_alloced()
        net.zero_grad()
        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
            optimizer.step()
        if(e%1==0):
            print_numpy("Log likelihood loss " ,torch.mean(distloss))
            print_numpy("Log determinant jacobian  " ,torch.mean(jacloss))
            print_numpy("Total loss ",loss)
#verify()
train()
#modelprint()

torch.cuda.synchronize()
del net
torch.cuda.synchronize()
torch.cuda.empty_cache()
torch.cuda.synchronize()
