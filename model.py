from transformer import *
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import ast
from numpy import load
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import torch
import random
from tqdm import tqdm

def compute_acc(pred_high,pred_low, ground_truth,threshold=0.08):
    diff_h = (torch.abs((pred_high - ground_truth[:,0]) / ground_truth[:,0])).sum()/pred_high.size(0)
    diff_l =(torch.abs((pred_low - ground_truth[:,1]) / ground_truth[:,1])).sum()/pred_low.size(0)

    return diff_h,diff_l

class Transformer(nn.Module):
    def __init__(self,n_blocks=2,d_model=64,n_heads=4,d_ff=256,dropout=0.2,vocab_size=28):
        super().__init__()
        self.emb = WordPositionEmbedding(vocab_size = vocab_size,d_model=d_model)
        self.decoder_emb = WordPositionEmbedding(vocab_size=vocab_size,d_model=d_model)
        self.encoder = TransformerEncoder(n_blocks=n_blocks,d_model=d_model,n_heads=n_heads,d_ff=d_ff,dropout=0.2)
        self.decoder = TransformerDecoder(n_blocks=n_blocks,d_model=d_model,d_feature=16,n_heads=n_heads,d_ff=d_ff,dropout=0.2)
    
    def forward(self,x):
        g = self.emb(x)
        encoded = self.encoder(g)
        p = self.decoder_emb(x)
        y = self.decoder(p, encoded)
        return y;



def evaluate(eval_model,epoch,criterion_high,criterion_low,data_source,dev='cpu'):
    eval_model.eval() # Turn on the evaluation mode
    count = 0
    with torch.no_grad():
        cum_loss_h = 0
        cum_loss_l = 0
        accs_h = 0
        accs_l = 0
        for batch in tqdm(data_source):
            data, targets = batch
            # targets = embs(targets.long())
            targets = targets.view(-1,2).to(dev)
            output = eval_model(data.to(dev))
            high = high.view(-1,1)
            low = low.view(-1,1)
            loss_high = criterion_high(high,targets[:,0])
            loss_low = criterion_low(low,targets[:,1])
            
            score_h,score_l = compute_acc(output,targets)
            accs_h += score_h
            accs_l += score_l
            # accs += ((torch.argmax(output,dim=1)==targets).sum().item()/output.size(0))
            cum_loss_h += loss_high.item()
            cum_loss_l += loss_low.item()
            count+=1
        print(epoch,"Loss high: ",(cum_loss_h/count),"Loss low: ",(cum_loss_l/count))
    return cum_loss_h/ (count), float(accs_h/(count)), float(accs_l/count)



    
   