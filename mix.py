import torch
def mixmat(size):
    m=torch.eye(size)
    m[1:,:]+=torch.eye(size)[:-1,:]
    m[:-1,:]+=torch.eye(size)[1:,:]
    return m
for i in range(1,100):
    try:
        #inv=torch.inverse(mixmat(i))
        print(torch.det(mixmat(i)))
        pass
    except:
        print(i)
