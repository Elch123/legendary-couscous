import sentencepiece as spm
import numpy as np
import pickle
import random
from hparams import hparams
#load the files and apply BPE encoding, this could be swapped for tokenization
from loader import makeendeprocessors
(bpemb_en,bpemb_de)=makeendeprocessors()
def loadtobpe(processor,filepath,start,end):
    lines=[]
    with open(filepath,newline="\n") as f:
        f=f.readlines()
        maxlen=len(f)
        for (count,line) in enumerate(f):
            if(count/maxlen>start and count/maxlen<end ): #
                data=processor.encode(line)
                lines.append(np.array(data))
                if(count%1000==0):
                    print(count)
    return lines
def makeslice(start,end,savefile):
    enbpe=loadtobpe(bpemb_en,"data/text.en",start,end)
    debpe=loadtobpe(bpemb_de,"data/text.de",start,end)
    print(len(enbpe))
    print(len(debpe))
    def sortbymaxlen(sentencepair):
        return max(len(sentencepair[0]),len(sentencepair[1]))
    #sort the sentece arrays by length
    #random.shuffle(enbpe[0])#Use an unstable sort for better mixed batches
    #random.shuffle(debpe[0])
    c=list(zip(enbpe,debpe))
    random.shuffle(c)
    en,de=zip(*c)
    enbpe,debpe=zip(*sorted(zip(en,de),key=sortbymaxlen))
    text=(enbpe,debpe)
    with open(savefile,'wb') as pairedtext:
        pickle.dump(text,pairedtext)
makeslice(.96,1,"validationdeen.pickle")
makeslice(.5,.7,"traindeen.pickle")
#print([enprocessor.DecodeIds(sentence) for sentence in enbpe[0:10]])
#print([deprocessor.DecodeIds(sentence) for sentence in debpe[0:10]])
