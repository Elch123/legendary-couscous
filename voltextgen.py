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
from tensorboardX import SummaryWriter
import json
from time import sleep
scale_max=300
"""with open("num_runs.json","w+") as f:
    f.write(json.dumps(0))"""
with open("num_runs.json","r+") as f:
    #print(f.read())
    run_count=json.loads(f.read())
    path='/tmp/estvolume'+str(run_count).zfill(4)
    print(path)
    writer = SummaryWriter(path)
    run_count+=1
    f.seek(0)
    f.write(json.dumps(run_count))
#tracemalloc.start()
sleep(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
r_change=4
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
optimizer = torch.optim.Adam(net.parameters(), lr=hparams['lr'])#SGD  momentum=hparams['momentum'], ,nesterov=False
optimizer = torch.optim.SGD(net.parameters(), lr=hparams['lr'],momentum=hparams['momentum'],nesterov=False)#SGD  , ,nesterov=False
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
def argmax_error_fn(batch,labels): #Assume labels are in 1 hot form
    print(batch.shape)
    print(labels.shape)
    selected=torch.sum(batch*labels,dim=1) #torch.index_select
    print(selected.shape)
    maxs=torch.max(batch*(1-labels),dim=1)
    error=maxs[0]-selected
    error=torch.max(error,dim=-1)
    print("error is " +str(error))
    return error[0]
def argmax_input_fn():
    b_size=hparams['batch_size']
    c=hparams['dim']
    length=2
    batch=torch.zeros(size=(b_size,c,length))
    for i in range(b_size):
        for j in range(length):
            chan=(torch.rand(1)*c).long()
            batch[i,chan,j]=1
    return batch
def argmax_mismatch(output,label):
    return argmax_error_fn(output,label)<0
def binary_input_fn():
    b_size=hparams['batch_size']
    c=hparams['dim']
    length=2
    batch=torch.zeros(size=(b_size,c,length))
    for i in range(b_size):
        #for j in range(length):
        if(torch.rand(1)>0.5):
            batch[i][0][0]=r_change+1
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
def find_root_batch(fn,start,direction,labels,start_scale=1e-2):
    #in_array=np.zeros(shape=(num_points,2))
    scales=torch.zeros(size=(start.shape[0],1,1), device=device)
    uppers=torch.zeros_like(scales, device=device)
    lowers=torch.zeros_like(scales, device=device)
    never=torch.zeros(size=(start.shape[0],), device=device)
    never[:]=1
    scales[:]=start_scale
    scales[:,0,0]*=torch.rand(scales.shape[0], device=device)+1
    #print("start scale" + str(start_scale))
    with torch.no_grad():
        out=fn(start).detach()
        initial_out=out
        #print(initial_out)
        for i in range(40):
            dists=direction*scales
            candidate_point=start+dists
            #print("evaling fn")
            out=fn(candidate_point)
            match=argmax_mismatch(out,labels)
            #print(scales[:,0,0])
            out_of_bounds=scales[:,0,0]>scale_max  #clip for accuracy in integration routine
            mismatched=(1-(1-match)*(1-out_of_bounds))#this is an OR operator Maybe use torch elementwise or if I can find it?
            #Must paralellize this slow code, or move to CPU, it's a bottleneck now!
            #This is a basic bisection root finding algorithm. It doubles the distance until the function changes sign, then starts bisecting.
            """for (j,val) in enumerate(mismatched):
                if(never[j]):
                    if(not val): #never=True,mismatch=False
                        scales[j]*=2
                    if(val): #never=True,mismatch=True
                        uppers[j]=scales[j]
                        lowers[j]=scales[j]/2
                        never[j]=0
                else:
                    if(val): #Never=False,mismatch=True
                        uppers[j]=scales[j]
                    else:
                        lowers[j]=scales[j] #Never=False,mismatch=False
                    scales[j]=(uppers[j]+lowers[j])/2 #Never=False"""
            mismatched=mismatched.float()
            #Paralell form: (Extememely ugly, but works because it hits every single case properly)
            #This is equivilent to the commented out routine above, read that to understatnd how this works
            uppers[:,0,0]=(never*(1-mismatched))*uppers[:,0,0]+(never*mismatched)*scales[:,0,0]+  (1-never)*mismatched*scales[:,0,0]+(1-never)*(1-mismatched)*uppers[:,0,0]
            lowers[:,0,0]=(never*(1-mismatched))*lowers[:,0,0]+(never*mismatched)*scales[:,0,0]/2+(1-never)*mismatched*lowers[:,0,0]+(1-never)*(1-mismatched)*scales[:,0,0]
            scales[:,0,0]=(never*(1-mismatched))*2*scales[:,0,0]+(1-never)*(uppers[:,0,0]+lowers[:,0,0])/2+(never*mismatched)*scales[:,0,0] #all 4 cases hit
            never=(1-never)*never+never*(1-mismatched)*never+never*mismatched*0
        dists=direction*scales
        candidate_point=start+dists
        return scales,candidate_point
def neg_log_p_batch(points,dist):
    dimentions=points.shape[-1]*points.shape[-2]
    area=logspherearea(torch.tensor(dimentions+0.0,device=device))
    sum_squared_terms=torch.sum(torch.sum(points**2,dim=-1),dim=-1)
    log_gaussian_density=-sum_squared_terms/2-dimentions/2*torch.log(torch.tensor(2*3.14159,device=device))
    log_volume=(dimentions-1)*torch.log(dist)
    log_density=log_gaussian_density+log_volume+area #These terms are multiplied, so add in the log space
    return log_density
def log_numerator_batch(points,dist):
    dimentions=points.shape[-1]*points.shape[-2]
    dimentions=torch.tensor(dimentions+0.0,device=device)
    area=logspherearea(dimentions)
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
def fast_basic_integrate_batch(fn,point,direction,dist,samples=6000):
    dim=point.shape[-1]*point.shape[-2]
    #print("dim is "+str(dim))
    point=torch.unsqueeze(point,dim=1)
    direction=torch.unsqueeze(direction,dim=1)
    eval_dists=torch.stack([torch.linspace(0,d.item(),steps=samples) for d in dist],dim=0).to(device)
    eval_dists_expanded=torch.unsqueeze(eval_dists,dim=-1)
    eval_dists_expanded=torch.unsqueeze(eval_dists_expanded,dim=-1)
    eval_points=point+direction*eval_dists_expanded
    #print("eval points shape is "+str(eval_points.shape))
    results=fn(eval_points,eval_dists)
    #print("results shape is " + str(results.shape))
    #print("dist is "+str(dist))
    eval_sums=logsumexp_batch(results)
    distlog=torch.log(torch.squeeze(dist))
    integral=eval_sums-torch.log(torch.tensor(samples+0.0,device=device))+distlog #Distlog is to multiply by distance integrating over,
    #eval_sums is the sum of all samples, subtract log of samples to divide by number of samples,
    #in this dimention to make everything add up to 1
    #print("integral shape is "+str(integral.shape))
    return integral
def integrals(center,direction,root):
    prob_integral=fast_basic_integrate_batch(neg_log_p_batch,center,direction,root[0])
    log_numerator_start=fast_basic_integrate_batch(log_numerator_batch,center,direction,root[0])
    end_p=neg_log_p_batch(root[1],torch.squeeze(root[0]))
    return (prob_integral,log_numerator_start,end_p)

def make_normals(fn,point,error_fn,labels): #The orientation of these normal vectors doesn't matter
    net.zero_grad()
    point.requires_grad=True
    with torch.enable_grad():
        output=torch.sum(error_fn(fn(point),labels)) #Derive normals using the gradient being normal to level surfaces
    output.backward()
    scaled_normals=point.grad
    normal_scales=magnitude_batch(scaled_normals)
    normal_scales=torch.reshape(normal_scales,(normal_scales.shape[0],1,1))
    normals=scaled_normals/normal_scales #Rescale the normals to have a length of one.
    net.zero_grad()
    return normals
def make_half_grad(fn,center,direction,labels):
    root=find_root_batch(fn,center,direction,labels)
    root_out_of_bounds=(root[0]>scale_max-.1).float() #Test to see if the root found was infinity
    normals=make_normals(fn,root[1],argmax_error_fn,labels)
    normals=root_out_of_bounds*direction+(1-root_out_of_bounds)*normals #Use the direction vector as the normal if it goes off to infinity.
    flat_direction=torch.reshape(direction,(direction.shape[0],1,-1))
    flat_normals=torch.reshape(normals,(normals.shape[0],-1,1))
    dotted=torch.matmul(flat_direction,flat_normals)
    (prob_integral,log_numerator_start,end_p)=integrals(center,direction,root)
    grad_end_mag=torch.exp(end_p-prob_integral)
    grad_end_mag*=torch.squeeze(dotted)
    grad_start_mag=torch.exp(log_numerator_start-prob_integral)
    #grad_start_mag*=torch.squeeze(dotted)
    return (root,grad_end_mag,grad_start_mag,prob_integral,normals)
def make_grad_batch(fn,center,labels,count=-1):
    global avggrad #Probably better to refactor this into a class, maybe do later
    delta=.01 #how for back to go for finite differences calculation
    direction=random_dir_vector_like(center)
    (roota,grad_end_mag_a,grad_start_mag_a,prob_integral_a,normals_a)=make_half_grad(fn,center,direction,labels)
    (rootb,grad_end_mag_b,grad_start_mag_b,prob_integral_b,normals_b)=make_half_grad(fn,center,-direction,labels)
    loss_ce=-torch.mean(prob_integral_a+prob_integral_b)/2
    stddev=torch.std(prob_integral_a+prob_integral_b)/2
    writer.add_scalar('Train/NLL', loss_ce, count)
    writer.add_scalar('Train/Loss Standard Deviation', stddev, count)
    print("mean log loss is "+str(loss_ce))
    grad_end_mag=torch.cat([grad_end_mag_a,grad_end_mag_b],dim=0)
    grad_end_mag=torch.reshape(grad_end_mag,(grad_end_mag.shape[0],1,1))
    grad_end=grad_end_mag*torch.cat([normals_a,normals_b]) #Use formula for graient at end
    writer.add_scalar('Train/Grad end', torch.mean(grad_end**2)**(1/2), count)
    #grad_start_mag=torch.cat([grad_start_mag_a,grad_start_mag_b],dim=0)
    grad_start_mag_a=torch.reshape(grad_start_mag_a,(grad_start_mag_a.shape[0],1,1))
    grad_start_mag_b=torch.reshape(grad_start_mag_b,(grad_start_mag_b.shape[0],1,1))
    #writer.add_scalar('Train/Grad start', torch.mean(grad_start_mag**2)**1/2, count)
    grad_start=-(grad_start_mag_a-grad_start_mag_b)*direction
    writer.add_scalar('Train/Grad start', torch.mean(grad_start**2)**1/2, count)
    end_points=torch.cat([roota[1],rootb[1]],dim=0)
    all_points=torch.cat([end_points,center],dim=0)
    all_grads=torch.cat([grad_end,grad_start],dim=0)
    writer.add_scalar('Train/Maximum grad', torch.max(all_grads), count)
    return (all_points,all_grads)
test_zero_point=torch.zeros(size=(10,8,6)).to(device)
#direction=random_dir_vector_like(test_zero_point)
#test_root=find_root_batch(test_eval_fn,test_zero_point,direction)
#test_integral=fast_basic_integrate_batch(test_zero_point,direction,test_root[0])
#print(test_integral)
#grad=make_grad_batch(test_eval_fn,test_zero_point)
#print(grad)
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
        target=argmax_input_fn()
        target=target.to(device)
        #print("chosen locations are " + str(target[:,0,0]))
        #verify_test(target)
        with torch.no_grad():
            start=net.inverse(target)[0]
        #print(target[:,0,0])
        #class_a_zero=torch.sum((target[:,0,0]==0).float())+.001
        #classadist=(torch.sum(magnitude_batch(start)*(target[:,0,0]==0).float()))/class_a_zero
        #class_b_zero=torch.sum((target[:,0,0]==r_change+1).float())+.001
        #classbdist=(torch.sum(magnitude_batch(start)*(target[:,0,0]==r_change+1).float()))/class_b_zero
        #print("class a distance is " +str(classadist))
        #print("center dists from zero are " +str(magnitude_batch(start)))
        #writer.add_scalar('Train/Class A dist', classadist, e)
        #writer.add_scalar('Train/Class B dist', classbdist, e)
        #start=start.permute(0,2,1)
        with torch.no_grad():
            (all_points,all_grads)=make_grad_batch(net,start,target,e)
            mags=magnitude_batch(reshaped_net(make_normal_batch_like(start)))
            #print("forward pass of data is "+str(mags))
            #outside_circle=mags>r_change
            #print("data outside circle "+str(outside_circle))
            #writer.add_scalar('Train/Points outside circle', torch.sum(outside_circle), e)
        with torch.no_grad():
            boundary_outs=net(all_points)
        boundary_ins=net.inverse(boundary_outs)[0]
        net.zero_grad()
        all_grads/=hparams['batch_size']
        all_grads/=hparams['dim']
        boundary_ins.backward(-all_grads)
        optimizer.step()
        if(e%20==0):
            print("chosen locations are " + str(target[:,0,0]))
            print("center dists from zero are " +str(magnitude_batch(start)))
            #print("forward pass of data is "+str(mags))
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
