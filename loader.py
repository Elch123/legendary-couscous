from bpemb import BPEmb
from hparams import hparams
def makeendeprocessors():
    dim=hparams['dim']
    vs=hparams['embed_symbols']
    bpemb_en = BPEmb(lang="en", dim=dim ,vs=vs)
    bpemb_de = BPEmb(lang="de", dim=dim ,vs=vs)
    return (bpemb_en,bpemb_de)
