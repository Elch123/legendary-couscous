from loader import makeendeprocessors
(bpemb_en,bpemb_de)=makeendeprocessors()
import torch
import torch.nn as nn

class BpEmbed(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams
        self.embed=torch.nn.Embedding(params['symbols'],params['num_hidden'])
