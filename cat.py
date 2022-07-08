from torch.utils.data import Dataset,DataLoader
from torch import nn
from tkinter import _flatten
from torchaudio.compliance.kaldi import fbank
import random
import numpy as np
import torchaudio
import torch


  
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size per GPU')
    parser.add_argument('--epochs', default=100, type=int, help='epochs')

class MyDataset(Dataset):
    def __init__(self, mode = 'train', batch_size = 8):
        print ("init dataset")
        self.mode = mode
        self.datafile = '../data/train_demo.list'
        self.file_ids = open(self.datafile,'r').readlines()
        self.labelids = np.array([int(line.strip().split(" ")[-1]) for line in self.file_ids])
        #get ids set for triplet loss
        self.ids = list(set(self.labelids))
         
        self.width = 500
        self.height = 64
        self.sample_rate = 8000
        self.batch_size = batch_size

    def __getitem__(self, batch_idx):
   
        files_per_id = 2
        
        assert self.batch_size % files_per_id == 0 
        #select ids in this batch for triplet loss
        selected_ids = random.sample(range(0,len(self.ids)), self.batch_size // files_per_id)
        selected_files = [ random.sample(range(np.where(self.labelids == selected_id)[0][0],np.where(self.labelids == selected_id)[0][-1] + 1 ),2) for selected_id in selected_ids] 
        selected_files = _flatten(selected_files)
        print ("selected_files:" + str(selected_files))
         
        features = None
        labels = None
        i = 0
        for idx in selected_files:

            f,label = self.file_ids[idx].strip().split(" ")
            wav,sr = torchaudio.load(f,normalize=False)
            assert sr == self.sample_rate
            wav = wav / 1.0
            feature = fbank(wav, dither=1,high_freq=-200, low_freq=64, htk_compat=True,  num_mel_bins=self.height, sample_frequency=self.sample_rate, use_energy=False, window_type='hamming')
            feature_len = len(feature)
            if self.mode == "train": #random start pieces
                rand_start = random.randint(0,feature_len - self.width)
                feature = feature[rand_start : rand_start + self.width]
            else: #fixed feature for test
                feature = feature[0 : self.width]

            #normalize
            std,mu = torch.std_mean(feature,dim=0)
            feature = (feature - mu) / (std + 1e-5)

            feature = torch.unsqueeze(feature, dim=0)
            label = torch.LongTensor([int(label)])
            if i == 0: 
                features = feature
                labels = label
            else:
                features = torch.cat((features,feature),0)
                labels = torch.cat((labels,label),0)
            i = i + 1

        #TODO: make sure feature is normalized.
        features = features.reshape(self.batch_size, self.width, self.height)

        return features,labels

    def __len__(self):
        return len(self.file_ids) // self.batch_size

class Generator():
    def __init__(self):
        print ("init Generator")
        
class Discriminator_speaker():
    def __init__(self):
        print ("init Discriminator_speaker")

class FullyConnect_speaker(nn.Module):
    def __init__(self):
        print ("init FullyConnect_speaker")
        super(FullyConnect_speaker,self).__init__() 
        self.fc == nn.Linear(4 * 512, 512) 

    def forward(self,x):
        return self.fc(x)

class Discriminator_channel():
    def __init__(self):
        print ("init Discriminator_channel")

def main(args):
    num_classes = 251
    batch_size = args.batch_size
    lr_init = args.lr
    epochs = args.epochs
    
    D1 = Discriminator_speaker().cuda()
    FC_D1 = FullyConnect_speaker().cuda()
    D2 = Discriminator_channel().cuda()
    G = Generator().cuda()
    
    Class_CrossEntropyLoss = nn.CrossEntropyLoss()
    AdversarialLoss = nn.BCELoss()
    
    for epoch in range(epochs):
        train_one_epoch(train_dataloader, G, D1, FC_D1, D2, Class_CrossEntropyLoss, AdversarialLoss, optimizer_G, optimizer_D1, optimizer_D2, epoch, batch_size, args)

def normalize(x, axis=1):
    x = 1.0*x / (torch.norm(x, 2, axis, keepdim=True) + 1e-12)
    return x

def train_one_epoch(train_dataloader, G, D1, FC_D1, D2, Class_CrossEntropyLoss, AdversarialLoss, optimizer_G, optimizer_D1, optimizer_D2, epoch, batch_size, args):
    for batch_id,(x,y) in enumerate(train_dataloader):
        x = x.cuda()
        y = y.cuda()
        
        generator_feature = G(x)
        spk_feature = D1(generator_feature)
        pred_speakerid = FC_D1(spk_feature)
        pred_channel = D2(generator_feature)

        Ls = Class_CrossEntropyLoss(pred_speakerid,y)
        #TODO: spk_feature needs to be normalized for cosine distance, hmm think.

        #Lt =  
                
        #compute loss
        #update gradients
        #save log & model
        pass
    
       
if __name__ == '__main__':
    #args = parse_args()
    #main(args)
    dataset = MyDataset()
    dataset.__getitem__(0)

