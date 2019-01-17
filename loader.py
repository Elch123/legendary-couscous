from bpemb import BPEmb
from hparams import hparams
def makeendeprocessors():
    dim=hparams['dim']
    bpemb_en = BPEmb(lang="en", dim=dim)
    bpemb_de = BPEmb(lang="de", dim=dim)
    return (bpemb_en,bpemb_de)
