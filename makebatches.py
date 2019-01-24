import numpy as np
import pickle
from loader import makeendeprocessors
(enprocessor,deprocessor)=makeendeprocessors()

def maxlen(langa,langb):
    return max(len(langa),len(langb))

class Batch_maker():
    def __init__(self,filename):
        with open(filename,'rb') as pairedtext:
            self.text=pickle.load(pairedtext)
            #print(self.text[0][100000])
    def maxlen(self,langa,langb):
        return max(len(langa),len(langb))

    def make_batch(self,maxsymbols):
        numstrings=len(self.text[1])
        strlen=1000
        while(strlen>100):
            topi=np.random.randint(numstrings)
            strlen=int(len(self.text[1][topi])*1.1+4)
        numback=max(maxsymbols//strlen,1)
        numback=numback+min(0,topi-numback)#clip number of elements going back if it is less than zero, to not overrun start of array. Watch that zero
        fronti=topi-numback
        batch=np.zeros(shape=(2,numback,strlen),dtype=np.int32)
        for i in range(numback):
            seta=self.text[0][fronti+i]
            maxtopa=min(len(seta),strlen)
            batch[0][i][0:maxtopa]=seta[0:maxtopa]
            setb=self.text[1][fronti+i]
            maxtopb=min(len(setb),strlen)
            batch[1][i][0:maxtopb]=setb[0:maxtopb]
            if(batch.shape[1]==0 or batch.shape[2]==0):
                return self.makebatch(maxsymbols)
        return batch
#b=Batch_maker("traindeen.pickle")
#print(b.make_batch(1000).shape)
