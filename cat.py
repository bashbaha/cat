from torch.utils.data import Dataset,DataLoader
from torch import nn
from tkinter import _flatten
from torchaudio.compliance.kaldi import fbank
from torch import optim
import random
import numpy as np
import torchaudio
import torch
import math
import argparse

  
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=2e-1, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size per GPU')
    parser.add_argument('--epochs', default=10, type=int, help='epochs')

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
        selected_files = [ random.sample(range(np.where(self.labelids == selected_id)[0][0],np.where(self.labelids == selected_id)[0][-1] + 1 ),files_per_id) for selected_id in selected_ids] 
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

class LSTMPCell(nn.Module):

    def __init__(self, input_size, hidden_size, projection_size):
        super(LSTMPCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.projection_size = projection_size

        self.weight_xi = nn.Parameter(torch.randn(hidden_size, input_size))
        self.weight_xf = nn.Parameter(torch.randn(hidden_size, input_size))
        self.weight_xc = nn.Parameter(torch.randn(hidden_size, input_size))
        self.weight_xo = nn.Parameter(torch.randn(hidden_size, input_size))

        self.bias_xi = nn.Parameter(torch.randn(hidden_size))
        self.bias_xf = nn.Parameter(torch.randn(hidden_size))
        self.bias_xc = nn.Parameter(torch.randn(hidden_size))
        self.bias_xo = nn.Parameter(torch.randn(hidden_size))

        self.weight_hi = nn.Parameter(torch.randn(hidden_size, projection_size))
        self.weight_hf = nn.Parameter(torch.randn(hidden_size, projection_size))
        self.weight_hc = nn.Parameter(torch.randn(hidden_size, projection_size))
        self.weight_ho = nn.Parameter(torch.randn(hidden_size, projection_size))

        self.bias_hi = nn.Parameter(torch.randn(hidden_size))
        self.bias_hf = nn.Parameter(torch.randn(hidden_size))
        self.bias_hc = nn.Parameter(torch.randn(hidden_size))
        self.bias_ho = nn.Parameter(torch.randn(hidden_size))

        self.weight_project = nn.Parameter(torch.randn(projection_size, hidden_size))


    def forward(self, input, state):
        if state is not None:
            hx, cx = state
        else:
            hx = input.new_zeros(input.size(0), self.projection_size, requires_grad=False)
            cx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)

        input_gate = torch.mm(input, self.weight_xi.t()) + self.bias_xi + torch.mm(hx, self.weight_hi.t()) + self.bias_hi
        forget_gate = torch.mm(input, self.weight_xf.t()) + self.bias_xf + torch.mm(hx, self.weight_hf.t()) + self.bias_hf
        cell_gate = torch.mm(input, self.weight_xc.t()) + self.bias_xc + torch.mm(hx, self.weight_hc.t()) + self.bias_hc
        output_gate = torch.mm(input, self.weight_xo.t()) + self.bias_xo + torch.mm(hx, self.weight_ho.t()) + self.bias_ho

        ct = torch.mul(forget_gate, cx) + torch.mul(input_gate, cell_gate)
        ht = torch.mul(output_gate, torch.tanh(ct))
        ht = torch.mm(ht, self.weight_project.t())

        return ht, (ht, ct)

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)

        nn.init.uniform_(self.weight_xi, -stdv, stdv)
        nn.init.uniform_(self.weight_xf, -stdv, stdv)
        nn.init.uniform_(self.weight_xc, -stdv, stdv)
        nn.init.uniform_(self.weight_xo, -stdv, stdv)

        nn.init.uniform_(self.weight_hi, -stdv, stdv)
        nn.init.uniform_(self.weight_hf, -stdv, stdv)
        nn.init.uniform_(self.weight_hc, -stdv, stdv)
        nn.init.uniform_(self.weight_ho, -stdv, stdv)

        nn.init.uniform_(self.bias_xi, -stdv, stdv)
        nn.init.uniform_(self.bias_xf, -stdv, stdv)
        nn.init.uniform_(self.bias_xc, -stdv, stdv)
        nn.init.uniform_(self.bias_xo, -stdv, stdv)

        nn.init.uniform_(self.bias_hi, -stdv, stdv)
        nn.init.uniform_(self.bias_hf, -stdv, stdv)
        nn.init.uniform_(self.bias_hc, -stdv, stdv)
        nn.init.uniform_(self.bias_ho, -stdv, stdv)

class LSTMPLayer(nn.Module):
    
    def __init__(self, input_size, hidden_size, projection_size):
        super(LSTMPLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.cell = LSTMPCell(input_size=input_size, hidden_size=hidden_size, projection_size=projection_size)
    
    def forward(self, input, state):
        outputs = []
        for item in range(input.size()[0]):
            out, state = self.cell(input[item],state)
            outputs += [out]
        return torch.stack(outputs, dim=0), state

class Generator(nn.Module):

    def __init__(self, input_size, hidden_size, projection_size):
        super(Generator, self).__init__()
        self.LSTMPLayer1 = LSTMPLayer(input_size=input_size, hidden_size=hidden_size, projection_size=projection_size)
        self.LSTMPLayer2 = LSTMPLayer(input_size=projection_size, hidden_size=hidden_size, projection_size=projection_size)

    def forward(self, input, state):
        #input:(timestep, batch_size, input_dimension)
        output, state = self.LSTMPLayer1(input, state)
        output, state = self.LSTMPLayer2(output, state)

        #output:(timestep, batch_size, hidden_size)
        return output
        
class Discriminator_speaker(nn.Module):

    def __init__(self, speaker_classes):
        super(Discriminator_speaker, self).__init__()
        self.speaker_classes = speaker_classes
        self.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding='same')
        self.maxpool1 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding='same')
        self.maxpool2 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding='same')
        self.maxpool3 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.conv4 = nn.Conv2d(256, 512, 3, stride=1, padding='same')
        self.maxpool4 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.conv5 = nn.Conv2d(512, 512, 3, stride=1, padding='same')
        self.avgpool = nn.AvgPool2d((31, 1), stride=1)
        self.fc = nn.Linear(4*512, speaker_classes)
    
    def forward(self, input):
        #input: (batch_size, 1, timestep_width, height)
        output = self.conv1(input)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.maxpool2(output)
        output = self.conv3(output)
        output = self.maxpool3(output)
        output = self.conv4(output)
        output = self.maxpool4(output)
        output = self.conv5(output)
        output = self.avgpool(output)
        output = output.squeeze()
        output = output.reshape(output.size(0),(output.size(1)*output.size(2)))
        #output = self.fc(output)
        #output: (batch_size, 4 x 512)   #TODO check
        return output
class Gradient_Reversal_Layer(nn.Module):

    def __init__(self, Lambda):
        super(Gradient_Reversal_Layer, self).__init__()
        self.Lambda = Lambda

    def forward(self, input):
        return input.view_as(input)

    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input * (-self.Lambda)

    def set_Lambda(self, Lambda):
        self.Lambda = Lambda



class Discriminator_channel(nn.Module):

    def __init__(self, GRL=False):
        super(Discriminator_channel, self).__init__()
        if GRL:
           self.grl = Gradient_Reversal_Layer(Lambda=1)
        self.fc1 = nn.Linear(64, 128)  #TODO check. input should be flatten, input dimension may be 500 x 64
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 8)
        self.fc6 = nn.Linear(8, 2)
        self.avgpool = nn.AvgPool2d(kernel_size=(500,1), stride=1)

    def forward(self, input):
        if getattr(self, "grl", None) is not None:
            input = self.grl(input)
        output = self.fc1(input)
        output = self.fc2(output)
        output = self.fc3(output)
        output = self.fc4(output)
        output = self.fc5(output)
        output = self.fc6(output)
        output = self.avgpool(output)
        output = output.squeeze()

        return output

class FullyConnect_speaker(nn.Module):
    def __init__(self):
        print ("init FullyConnect_speaker")
        super(FullyConnect_speaker,self).__init__() 
        self.fc = nn.Linear(4 * 512, 512) 

    def forward(self,x):
        return self.fc(x)

class TripleLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripleLoss, self).__init__()
        self.margin = margin # 阈值
        self.rank_loss = nn.MarginRankingLoss(margin=margin)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.tripletloss = nn.TripletMarginLoss(margin=1.0, reduction='sum')

    def forward(self, inputs, labels, norm=False):      
        dist_mat = self.cosine_dist(inputs, inputs, norm=norm)  # 距离矩阵,越大越相似。
        dist_ap, dist_an = self.hard_sample(dist_mat, labels) # 取出每个anchor对应的hard sample.
        loss = self.tripletloss(inputs,dist_ap,dist_an)
        
        return loss

    @staticmethod
    def hard_sample( dist_mat, labels, ):
        # 距离矩阵的尺寸是 (batch_size, batch_size)
        assert len(dist_mat.size()) == 2
        assert dist_mat.size(0) == dist_mat.size(1)
        N = dist_mat.size(0)

        # 选出所有正负样本对
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t()) # 两两组合， 取label相同的a-p
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t()) # 两两组合， 取label不同的a-n

        list_ap, list_an = [], []
        # 取出所有正样本对和负样本对的距离值
        for i in range(N):
            list_ap.append( dist_mat[i][is_pos[i]].min().unsqueeze(0) )  #hard: 相同标签，选择cosine距离小的。
            list_an.append( dist_mat[i][is_neg[i]].max().unsqueeze(0) )  #hard: 不同标签，选择cosine距离大的。

        dist_ap = torch.cat(list_ap)  # 将list里的tensor拼接成新的tensor
        dist_an = torch.cat(list_an)

        return dist_ap, dist_an

    @staticmethod
    def normalize(x, axis=1):
        x = 1.0*x / (torch.norm(x, 2, axis, keepdim=True) + 1e-12)
        return x

    @staticmethod
    def euclidean_dist(x, y, norm=True):
        if norm: #when vector is normalized, euclidean_dist is almost equal to cosine dist.
            x = self.normalize(x)
            y = self.normalize(y)
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist = xx + yy # 任意的两个样本组合， 求第二范数后求和 x^2 + y^2
        dist.addmm_( 1, -2, x, y.t() ) # (x-y)^2 = x^2 + y^2 - 2xy
        dist = dist.clamp(min=1e-12).sqrt() #开方
        return dist

    @staticmethod
    def cosine_dist(x, y):
        x = self.normalize(x)
        y = self.normalize(y)
        dist = torch.mm(x,y) #cosine distance because x,y are normalized.
        return dist



def normalize(x, axis=1):
    x = 1.0*x / (torch.norm(x, 2, axis, keepdim=True) + 1e-12)
    return x

def main(args):
    num_classes = 20
    batch_size = 8
    lr_init = 0.02
    epochs = 10
    input_size = 64 
    hidden_size = 128
    projection_size = 64
    speaker_classes = 512
    
    D1 = Discriminator_speaker(speaker_classes=num_classes).cuda()
    FC_D1 = FullyConnect_speaker().cuda()
    D2 = Discriminator_channel().cuda()
    G = Generator(input_size=input_size, hidden_size=hidden_size, projection_size=projection_size).cuda()
    
    Class_CrossEntropyLoss = nn.CrossEntropyLoss()
    AdversarialLoss = nn.BCELoss()
    Triple_loss = TripleLoss()

    optimizer_D1G = optim.SGD([ {'params': D1.parameters()}, {'params': FC_D1.parameters()}, {'params': G.parameters(), 'lr': 1e-3} ], lr=2e-1, momentum=0.9)
    optimizer_D2 = optim.SGD([ {'params': D2.parameters()}, ], lr=2e-1, momentum=0.9)


    train_dataset = MyDataset(mode='train')
    train_dataloader = DataLoader(train_dataset,batch_size = 1, shuffle=False, drop_last=True, num_workers=5 )
    
    for epoch in range(epochs):
        train_one_epoch(train_dataloader, G, D1, FC_D1, D2, Class_CrossEntropyLoss, AdversarialLoss,Triple_loss,  optimizer_D1G, optimizer_D2, epoch, batch_size, args)


def train_one_epoch(train_dataloader, G, D1, FC_D1, D2, Class_CrossEntropyLoss, AdversarialLoss, Triple_loss,  optimizer_D1G, optimizer_D2, epoch, batch_size, args):
    for batch_id,(x,y) in enumerate(train_dataloader):

        optimizer_D1G.zero_grad()
        optimizer_D2.zero_grad()

        x = x.cuda()
        y = y.cuda()
        input_size = 64 
        hidden_size = 128
        projection_size = 64
        speaker_classes = 512
        
        #example input & output
        input = torch.randn(8, 500, 64, 1)
        input = input.squeeze(dim=-1)
        input = input.transpose(dim0=0, dim1=1)
        hx = input.new_zeros(input.size(1), projection_size, requires_grad=False)
        cx = input.new_zeros(input.size(1), hidden_size, requires_grad=False)
        state = [hx, cx]
        print (x.shape)
        generator_feature = G(x,state)
        print (generator_feature.shape)
        spk_embedding = D1(generator_feature)
        pred_speakerid = FC_D1(spk_embedding)
        pred_channel = D2(generator_feature)
        
 
        #train G,D1
        G.train()
        D1.train()
        D2.eval()
        #compute loss
        Ls = Class_CrossEntropyLoss(pred_speakerid,y)
        Lt = Triple_loss(spk_embedding, y)
        L_d1 = Ls + Lt
        L_d1.backward()
        #update gradients.  TODO:MIN?
        optimizer_D1G.step()

        #train D2
        G.eval()
        D1.eval()
        D2.train()
        #get channel_label
        channel_label = []
        for i in y:
            if i.item() < 10:
                channel_label.append(0)    
            else:
                channel_label.append(1)    
        channel_label = torch.from_numpy(np.array(channel_label))
        L_d2 = Class_CrossEntropyLoss(pred_channel,channel_label)
        L_d2.backward()
        #update gradients. TODO:max?
        optimizer_D2.step()
        

        # example input & output
        # input = torch.randn(2, 500, 64, 1)
        # input = input.squeeze(dim=-1)
        # input = input.transpose(dim0=0, dim1=1)
        # hx = input.new_zeros(input.size(1), projection_size, requires_grad=False)
        # cx = input.new_zeros(input.size(1), hidden_size, requires_grad=False)
        # state = [hx, cx]
        # G = Generator(input_size=input_size, hidden_size=hidden_size, projection_size=projection_size)
        # output = G(input, state)
        # output_LSTM = output.transpose(dim0=0, dim1=1)
        # output = output_LSTM.unsqueeze(dim=1)
        # D = Discriminator_speaker(speaker_classes=speaker_classes)
        # output = D(output)
        # D2 = Discriminator_channel(GRL=True)
        # output = D2(output_LSTM)

        pass
    
       
if __name__ == '__main__':
    args = parse_args()
    main(args)
    #dataset = MyDataset()
    #dataset.__getitem__(0)

