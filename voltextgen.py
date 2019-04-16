from loader import makeendeprocessors
(bpemb_en,bpemb_de)=makeendeprocessors()
import torch
import torch.nn as nn
import torch.nn.functional as F
from volhparams import hparams
import numpy as np
from embed import BpEmbed
from makebatches import Batch_maker
from blocks import Net
#tracemalloc.start()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
r_change=1
print(device)
#engbpe=BpEmbed(hparams,bpemb_en)
#debpe=BpEmbed(hparams,bpemb_de)
#maker=Batch_maker("traindeen.pickle")
net=Net(hparams)
net=net.to(device)
"""
Updating the net for translation
figure out how to wire the parts together
translate!
"""
optimizer = torch.optim.SGD(net.parameters(), lr=hparams['lr'], momentum=hparams['momentum'],nesterov=False)
def reshaped_net(batch):
    net_in=batch.permute(0,2,1)
    result=net(net_in).permute(0,2,1)
    return result
def make_normal_batch(batch_size,channels,seqlen):
    data=torch.randn(batch_size*channels*seqlen).reshape((batch_size,channels,seqlen))
    return data
def make_normal_batch_like(b):
    return make_normal_batch(b.shape[0],b.shape[1],b.shape[2]).to(b.device)
def make_batch(batch_size):
    batch=maker.make_batch(batch_size) #English=0 not German=1
    target=engbpe(torch.tensor(batch[0]).long())
    #target=blur_batch(target)
    target=target.permute(0,2,1)
    source=debpe(torch.tensor(batch[1]).long()).permute(0,2,1)
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
def logspherearea(dim):
    return torch.log(torch.tensor(2.0,device=device))+torch.log(torch.tensor(3.14159,device=device))*dim/2-torch.lgamma(dim/2)
def magnitude_batch(vector):
    return torch.sum(torch.sum(vector**2,-1),-1)**(1/2) #Sum all dim except 0
def test_input_fn(batch):
    return make_normal_batch(hparams['batch_size'],8,6)
def binary_input_fn():
    b_size=hparams['batch_size']
    c=hparams['dim']
    length=8
    batch=torch.zeros(size=(b_size,c,length))
    for i in range(b_size):
        #for j in range(length):
        if(torch.rand(1)>0.5):
            batch[i][0][0]=r_change+1
    return batch
def test_eval_fn(batch):
    return batch
def random_dir_vector_like(batch):
    x=torch.randn(batch.numel(), device=device).reshape(batch.shape)
    x=x/magnitude_batch(x).reshape((batch.shape[0],1,1))
    return x
def mismatch(outa,outb):
    #print(outa.shape)
    da=magnitude_batch(outa)
    db=magnitude_batch(outb)

    init=da>r_change #Within unit circle
    now=db>r_change #outside unit circle
    changed=(init!=now)
    #print(changed)
    return changed
def find_root_batch(fn,start,direction,start_scale=1e-2):
    #in_array=np.zeros(shape=(num_points,2))
    scales=torch.zeros(size=(start.shape[0],1,1), device=device)
    uppers=torch.zeros_like(scales, device=device)
    lowers=torch.zeros_like(scales, device=device)
    never=torch.zeros(size=(start.shape[0],), device=device)
    never[:]=1
    scales[:]=start_scale
    #print("start scale" + str(start_scale))
    with torch.no_grad():
        out=fn(start).detach()
        initial_out=out
        #print(initial_out)
        for i in range(40):
            #print(start.shape)
            #print(direction.shape)
            #print(scales.shape)
            dists=direction*scales
            candidate_point=start+dists
            print("evaling fn")
            out=fn(candidate_point)
            print("other stuff\n")
            #print(candidate_point.shape)
            #print(out)
            match=mismatch(initial_out,out)
            #print(scales[:,0,0])
            out_of_bounds=scales[:,0,0]>200  #clip for accuracy in integration routine
            match=(1-(1-match)*(1-out_of_bounds))#this is an OR operator Maybe use torch elementwise or if I can find it?

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
        dists=direction*scales
        candidate_point=start+dists
        return scales,candidate_point
def neg_log_p_batch(points,dist):
    dimentions=points.shape[-1]*points.shape[-2]
    area=logspherearea(torch.tensor(dimentions+0.0,device=device))
    #print("dims are "+str(dimentions))
    sum_squared_terms=torch.sum(torch.sum(points**2,dim=-1),dim=-1)
    log_gaussian_density=-sum_squared_terms/2-dimentions/2*torch.log(torch.tensor(2*3.14159,device=device))
    log_volume=(dimentions-1)*torch.log(dist)
    log_density=log_gaussian_density+log_volume+area #These terms are multiplied, so add in the log space
    return log_density
def log_numerator_batch(points,dist):
    dimentions=points.shape[-1]*points.shape[-2]
    dimentions=torch.tensor(dimentions+0.0,device=device)
    area=logspherearea(dimentions)
    #print("dims are "+str(dimentions))
    sum_squared_terms=torch.sum(torch.sum(points**2,dim=-1),dim=-1)
    log_gaussian_density=-sum_squared_terms/2-dimentions/2*torch.log(torch.tensor(2*3.14159,device=device))
    log_volume=(dimentions-2)*torch.log(dist)
    log_pascal_mult=torch.log(dimentions-1)
    log_numerator=log_gaussian_density+log_volume+area+log_pascal_mult #These terms are multiplied, so add in the log space
    return log_numerator
def logsumexp_batch(tensor): #implements logsumexp trick for numerically stable adding of logarithms
    maximum=torch.max(tensor,dim=-1)[0] #The imputs to this fn should only be 2 dim array,
    #one for batch, and one for values to logsumexp so no need to max over more than 1 dim
    tensor-=torch.unsqueeze(maximum,dim=-1)
    remaider_log_sum=torch.log(torch.sum(torch.exp(tensor),dim=-1))
    result=remaider_log_sum+maximum
    return result
def fast_basic_integrate_batch(fn,point,direction,dist,samples=3000):
    dim=point.shape[-1]*point.shape[-2]
    #print("dim is "+str(dim))
    point=torch.unsqueeze(point,dim=1)
    direction=torch.unsqueeze(direction,dim=1)
    eval_dists=torch.stack([torch.linspace(0,d.item(),steps=samples) for d in dist],dim=0).to(device)
    eval_dists_expanded=torch.unsqueeze(eval_dists,dim=-1)
    eval_dists_expanded=torch.unsqueeze(eval_dists_expanded,dim=-1)
    #print("points shape is "+str(point.shape))
    #print("directions shape is "+str(direction.shape))
    #print("dists shape is "+str(eval_dists_expanded.shape))
    eval_points=point+direction*eval_dists_expanded
    #print("eval points shape is "+str(eval_points.shape))
    results=fn(eval_points,eval_dists)
    #print("results shape is " + str(results.shape))
    #print("dist is "+str(dist))
    eval_sums=logsumexp_batch(results)
    #print("eval sums shape is "+str(eval_sums.shape))
    #print("eval sums is "+str(eval_sums))
    distlog=torch.log(torch.squeeze(dist))
    integral=eval_sums-torch.log(torch.tensor(samples+0.0,device=device))+distlog #Distlog is to multiply by distance integrating over,
    #eval_sums is the sum of all samples, subtract log of samples to divide by number of samples,
    #in this dimention to make everything add up to 1
    #print("integral shape is "+str(integral.shape))
    return integral
def make_grad_batch(fn,center):
    delta=.01 #how for back to go for finite differences calculation
    direction=random_dir_vector_like(center)
    #print("random direction shape is " + str(direction.shape))
    root=find_root_batch(fn,center,direction) #magnitude_batch, and point of intersection
    print("root distance is "+str(torch.squeeze(root[0])))
    #print("root distance from zero is "+str(magnitude_batch(root[1])))
    #print("root  is " + str(root))
    prob_integral=fast_basic_integrate_batch(neg_log_p_batch,center,direction,root[0])
    #prob_integral_start_shifted=fast_basic_integrate_batch(neg_log_p_batch,center-direction*delta,direction,root[0]+delta) #compute gradient of first point using finite differences
    #prob_integral_end_shifted=fast_basic_integrate_batch(neg_log_p_batch,center,direction,root[0]+delta) #compute gradient of last point using finite differences
    log_numerator_start=fast_basic_integrate_batch(log_numerator_batch,center,direction,root[0])
    #print("log numerator start is "+str(log_numerator_start))
    #Could also revert to using fundemental therom of calc 4 last point
    #print(root[1].shape)
    #print(root[0].shape)
    end_p=neg_log_p_batch(root[1],torch.squeeze(root[0]))
    #print("end p is " + str(end_p))

    print("logarithm of prob integral is " + str(prob_integral))
    #print("loss is "+str(-torch.mean(prob_integral)))
    #print("logarithm of prob integral two is " + str(prob_integral_start_shifted))
    #finite_grad_end_mag=torch.tensor(prob_integral_end_shifted-prob_integral)/delta
    grad_end_mag=torch.exp(end_p-prob_integral)
    #print("diff between grad end methods is " + str(grad_end_mag-finite_grad_end_mag))
    grad_end_mag=torch.reshape(grad_end_mag,(grad_end_mag.shape[0],1,1))
    print("gradient magnitude at end is " + str(torch.squeeze(grad_end_mag)))
    grad_end=grad_end_mag*direction #Use formula for graient at end
    #grad_start=(torch.exp(torch.tensor(prob_integral_start_shifted))-torch.exp(torch.tensor(prob_integral)))/delta*-direction
    #finite_grad_start_mag=torch.tensor(prob_integral_start_shifted-prob_integral)/delta
    grad_start_mag=torch.exp(log_numerator_start-prob_integral)
    #print("diff between grad start methods is " + str(grad_start_mag-finite_grad_start_mag))
    grad_start_mag=torch.reshape(grad_start_mag,(grad_start_mag.shape[0],1,1))
    print("gradient magnitude at start is " + str(torch.squeeze(grad_start_mag)))
    grad_start=-grad_start_mag*direction #derivative of negative log of likelihood w/ finite differences. Negative, because in opposite direction of random vector
    #print("direction is now " + str(direction))
    #print("grad start is " + str(grad_start))

    return (root[1],grad_end,center,grad_start)
test_zero_point=torch.zeros(size=(10,8,6)).to(device)
#direction=random_dir_vector_like(test_zero_point)
#test_root=find_root_batch(test_eval_fn,test_zero_point,direction)
#test_integral=fast_basic_integrate_batch(test_zero_point,direction,test_root[0])
#print(test_integral)
grad=make_grad_batch(test_eval_fn,test_zero_point)
print(grad)
def verify_test(batch):
    with torch.no_grad():
        print()
        passed=net(net.inverse(batch)[0])
        print("")
        passedtwo=net.inverse(net(batch))[0]
        erra=torch.mean(batch-passed)
        errb=torch.mean(batch-passedtwo)
    print_numpy("Inverse first error ",erra)
    print_numpy("Forward first error ",errb)
def train():
    for e in range(hparams['batches']):
        if(e%40==20):
            #prof_forward()
            #verify()
            #modelprint()
            pass
        #target=make_batch(hparams['batch_size'])
        target=binary_input_fn()
        target=target.to(device)
        print("chosen locations are " + str(target[:,0,0]))
        #verify_test(target)
        with torch.no_grad():
            start=net.inverse(target)[0]
        print("center dists from zero are " +str(magnitude_batch(start)))
        start=start.permute(0,2,1)
        grads=make_grad_batch(reshaped_net,start)
        with torch.no_grad():
            mags=magnitude_batch(reshaped_net(make_normal_batch_like(start)))
            print("forward pass of data is "+str(mags))
            print("data outside circle "+str(mags>r_change))
        all_points=torch.cat([grads[0],grads[2]],dim=0)
        all_grads=torch.cat([grads[1],grads[3]],dim=0)
        with torch.no_grad():
            boundary_outs=net(all_points.permute(0,2,1))
        boundary_ins=net.inverse(boundary_outs)[0]
        net.zero_grad()
        all_grads/=hparams['batch_size']
        all_grads/=hparams['dim']
        boundary_ins.backward(-all_grads.permute(0,2,1))
        optimizer.step()
        print("")
#for i in range(100):
#    print(logspherearea(torch.tensor(i).float()))
#verify()
train()
#modelprint()

torch.cuda.synchronize()
del net
torch.cuda.synchronize()
torch.cuda.empty_cache()
torch.cuda.synchronize()
