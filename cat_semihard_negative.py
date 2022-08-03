from torch.utils.data import Dataset,DataLoader
from torch import nn
from tkinter import _flatten
from torchaudio.compliance.kaldi import fbank
from torch import optim
import random
import numpy as np
import torchaudio
import torch
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)
import math
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
writer = SummaryWriter(log_dir='log_semihardNegative_760')

#change dataset, you may need to modify:
#1.batch_size
#2.pos_files_per_id
#3.classes_weight
#4.channel_weight
#5.num_classes
#6.datafile
#7.channel_label
  
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size per GPU')
    parser.add_argument('--num_classes', default=760, type=int, help='batch size per GPU')
    parser.add_argument('--epochs', default=100, type=int, help='epochs')
    parser.add_argument('--margin', default=0.3, type=float, help='margin for triplet loss')
    args = parser.parse_args()
    return args

class MyDataset(Dataset):
    def __init__(self, mode = 'train', batch_size = 8):
        #print ("init dataset")
        self.mode = mode
        self.datafile = '../data/train.760.list'
        self.file_ids = open(self.datafile,'r').readlines()
        self.labelids = np.array([int(line.strip().split(" ")[-1]) for line in self.file_ids])
        #get ids set for triplet loss
        self.ids = list(set(self.labelids))
         
        self.width = 500
        self.height = 64
        self.sample_rate = 8000
        self.batch_size = batch_size

    def __getitem__(self, batch_idx):
   
        pos_files_per_id = 2 #max is 4 for 95 self data.
        
        assert self.batch_size % pos_files_per_id == 0 
        #select ids in this batch for triplet loss
        selected_ids = random.sample(range(0,len(self.ids)), self.batch_size // pos_files_per_id)
        selected_files = [ random.sample(range(np.where(self.labelids == selected_id)[0][0],np.where(self.labelids == selected_id)[0][-1] + 1 ),pos_files_per_id) for selected_id in selected_ids] 
        selected_files = _flatten(selected_files)
        #print ("selected_files:" + str(selected_files))
         
        features = None
        labels = None
        i = 0
        for idx in selected_files:

            f,label = self.file_ids[idx].strip().split(" ")
            #print ('read from :' + str(f))
            wav,sr = torchaudio.load(f,normalize=True)
            assert sr == self.sample_rate
            wav = wav / 1.0
            feature = fbank(wav, dither=1,high_freq=-200, low_freq=64, htk_compat=True,  num_mel_bins=self.height, sample_frequency=self.sample_rate, use_energy=False, window_type='hamming', subtract_mean=True)
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
        print ('input_features:')
        print (features.shape)
        #print (features)

        return features,labels

    def __len__(self):
        return len(self.file_ids) // self.batch_size


class Generator(nn.Module):
    #use official LSTMP
    def __init__(self,input_size, hidden_size, projection_size, batch_size = 8):
        super(Generator, self).__init__()
        self.num_layers = 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lstmp = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=self.num_layers, proj_size=projection_size)

    def forward(self,x):
        #h_0 = torch.randn(self.num_layers, self.batch_size, self.input_size).cuda()
        #c_0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size).cuda()

        #output, (h,c) = self.lstmp(x, (h_0, c_0))
        output, (h,c) = self.lstmp(x)
        return output
        
class Discriminator_speaker(nn.Module):

    def __init__(self, num_classes):
        super(Discriminator_speaker, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding='same')
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.tanh = nn.Tanh()
        #TODO need dorpout?
        self.maxpool1 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout= nn.Dropout(p=0.2)
        self.maxpool2 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding='same')
        self.bn3 = nn.BatchNorm2d(256)
        self.maxpool3 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.conv4 = nn.Conv2d(256, 512, 3, stride=1, padding='same')
        self.bn4 = nn.BatchNorm2d(512)
        self.maxpool4 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.conv5 = nn.Conv2d(512, 512, 3, stride=1, padding='same')
        self.bn5 = nn.BatchNorm2d(512)
        self.avgpool = nn.AvgPool2d((31, 4), stride=1) # TODO check. diff with paper
        self.fc = nn.Linear(512, self.num_classes) 

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight, gain= nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.conv2.weight, gain= nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.conv3.weight, gain= nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.conv4.weight, gain= nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.conv5.weight, gain= nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.fc.weight, gain= nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.conv3.bias)
        nn.init.zeros_(self.conv4.bias)
        nn.init.zeros_(self.conv5.bias)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x):
        #input: (batch_size, 1, timestep_width, height)
        #TODO find the best compose of relu and tanh.
        output = self.conv1(x)
        print ('output_conv1')
        print (output.shape)
        #print (output)
        output = self.tanh(output)
        print ('output_tanh1')
        print (output.shape)
        #print (output)
        output = self.bn1(output)
        print ('output_bn1')
        #print (output.shape)
        #print (output)
        output = self.maxpool1(output)
        print ('output_maxpool1')
        #print (output)
        output = self.bn2(self.tanh(self.conv2(output)))
        print ('output_bn2')
        #print (output)
        output = self.maxpool2(output)
        print ('output_maxpool2')
        #print (output)
        output = self.bn3(self.tanh(self.conv3(output)))
        print ('output_bn3')
        #print (output)
        output = self.maxpool3(output)
        print ('output_maxpool3')
        #print (output)
        output = self.bn4(self.tanh(self.conv4(output)))
        print ('output_bn4')
        #print (output)
        output = self.maxpool4(output)
        print ('output_maxpool4')
        #print (output)
        output = self.bn5(self.tanh(self.conv5(output)))
        print ('output_bn5')
        #print (output)
        output = self.avgpool(output)
        print ('output_avgpool')
        #print (output)
        print ('-------------------')
        print ('Discriminator_speaker output.shape')
        embedding = output.reshape(-1,512)
        class_logit = self.fc(embedding)

        #output: (batch_size, 512)   #TODO check
        return embedding, class_logit

class FullyConnect_speaker(nn.Module):
    def __init__(self, num_classes):
        #print ("init FullyConnect_speaker")
        super(FullyConnect_speaker,self).__init__() 
        self.num_classes = num_classes
        self.fc = nn.Linear(512, self.num_classes) 
        self.tanh = nn.Tanh()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight, gain= nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.fc.bias)

    def forward(self,x):
        print ('---------------------')
        print ('FullyConnect_speaker input.shape')
        print (x.shape)
        output = self.tanh(self.fc(x))
        #print (output.shape)
        print ('---------------------')
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

    def __init__(self):
        super(Discriminator_channel, self).__init__()
        self.grl = Gradient_Reversal_Layer(Lambda=1)
        #TODO check. may need dropout , RELU or Sigmoid
        self.fc1 = nn.Linear(500 * 64, 128)  #TODO check. input should be flatten, input dimension may be (batch_size, 500 x 64)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout= nn.Dropout(p=0.2)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128,64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, 8)
        self.bn5 = nn.BatchNorm1d(8)
        self.fc6 = nn.Linear(8, 4)
        self.bn6 = nn.BatchNorm1d(4)
        self.avgpool = nn.AvgPool1d(kernel_size=2, stride=2) #TODO check.

    def forward(self, input):
        input = self.grl(input)
        #print ('---------------------')
        #print ('Discriminator_channel input.shape')
        #print (input.shape)
        input = input.reshape(-1,500 * 64)
        #print (input.shape)
        output = self.fc1(input)
        print ('---------------------')
        print ('Discriminator_channel output.shape')
        print (output.shape)
        output = self.dropout(self.tanh(self.bn1(self.fc1(input))))
        output = self.dropout(self.tanh(self.bn2(self.fc2(output))))
        output = self.dropout(self.tanh(self.bn3(self.fc3(output))))
        output = self.dropout(self.tanh(self.bn4(self.fc4(output))))
        output = self.dropout(self.tanh(self.bn5(self.fc5(output))))
        output = self.bn6(self.fc6(output))
        output = output.unsqueeze(1)
        output = self.avgpool(output)
        #print (output.shape)
        output = output.squeeze()
        #print (output.shape)
        #print ('---------------------')

        return output

        

class TripleLoss(nn.Module):
    def __init__(self):
        super(TripleLoss, self).__init__()
        #self.margin = margin # 阈值
        #self.rank_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, labels, norm=False):      
        #inputs should be normalized.
        #inputs = normalize(inputs)
        dist_mat = self.cosine_dist(inputs, inputs)  # 距离矩阵,越大越相似。
        #dist_ap, dist_an = self.hard_sample(dist_mat, labels) # 取出每个anchor对应的hard sample.
        #print ('TripleLoss loss1:') 
        #print (loss)
        #print ('dist_an:')
        #print (dist_an)
        #print ('dist_ap:')
        #print (dist_ap)
        #loss = nn.functional.relu(loss)
        #print ('TripleLoss loss2:') 
        #print (loss)
        #loss = torch.sum(loss)
        #loss = dist_an + self.margin - dist_ap


        id_ap, id_an = self.hard_sample(dist_mat, labels) # 取出每个anchor对应的hard sample.
        print ('official id_ap:')
        #print (id_ap)
        print ('official id_an:')
        #print (id_an)
        #loss = self.official_tripletloss(inputs, inputs[id_ap], inputs[id_an])
        #return loss
        return inputs[id_ap], inputs[id_an]

    @staticmethod
    def hard_sample( dist_mat, labels):
        # 距离矩阵的尺寸是 (batch_size, batch_size)
        assert len(dist_mat.size()) == 2
        assert dist_mat.size(0) == dist_mat.size(1)
        N = dist_mat.size(0)
        print ('---------------------')
        print ('TripleLoss dist_mat')
        print (dist_mat[0:20])
        print (dist_mat[-20:])
        dist_mat_ap = dist_mat.clone()
        dist_mat_an = dist_mat.clone()
        

        # 选出所有正负样本对
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t()) # 两两组合， 取label相同的a-p
        for i in range(N):
            is_pos[i][i] = False

        print ('TripleLoss dist_mat_ap init')
        #print (dist_mat_ap)
        for i in range(N):
            dist_mat_ap[i][~is_pos[i]] += 2
        print ('TripleLoss dist_mat_ap changed')
        #print (dist_mat_ap)
        
        id_ap = torch.argmin(dist_mat_ap,1)
        #print ('list_ap position:')
        #print (id_ap)
        #list_ap = torch.amin(dist_mat_ap,1)
        #list_an = torch.amax(dist_mat_an,1)
        #return list_ap, list_an
        
                
        
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t()) # 两两组合， 取label不同的a-n
        print ('TripleLoss dist_mat_an init')
        #print (dist_mat_an)

        for i in range(N):
            #for hardest sample
            #dist_mat_an[i][~is_neg[i]] -= 2 
            #for easyest sample
            dist_mat_an[i][~is_neg[i]] += 2 

        print ('TripleLoss dist_mat_an changed')
        #print (dist_mat_an)

        #TODO use semi-hardest sample
        #id_an = torch.argmax(dist_mat_an,1) #hardest sample
        id_an = torch.argmin(dist_mat_an,1) #easyest sample
        #print ('list_an position:')
        print ('easyest id_an:')
        #print (id_an)

        #after easyest sample, then select semi-hard negative vector, the compare these two.
        #mean_ap = dist_mat_ap[is_pos].reshape(N,-1).mean(axis=1)
        min_ap = dist_mat_ap[is_pos].reshape(N,-1).amin(axis=1)

        #use fixed adjust
        adjust = 1
        #use random adjust 
        #adjust = torch.randint(low=1,high=10,size=(1,1))[0][0].item()
        print ('adjust:')
        print (adjust)

        print ('min_ap adjust before')
        #print (min_ap)
        min_ap[min_ap.gt(0)] /= adjust
        min_ap[min_ap.lt(0)] *= adjust
        #print ('mean_ap')
        #print (mean_ap)
        print ('min_ap adjust after')
        #print (min_ap)

        for i in range(N):
            dist_mat_an[i][~is_neg[i]] -= 4 #+2-4 = -2
            #print ('dist_mat_an -5 before: ' + str(i))
            #print (dist_mat_an[i])
            #print ('dist_mat_an[i].gt(min_ap[i])')
            #print (min_ap[i])
            #print (dist_mat_an[i].gt(min_ap[i]))
            dist_mat_an[i][dist_mat_an[i].gt(min_ap[i]) ] -= 5 #important
            #print ('dist_mat_an -5 after: ' + str(i))
            #print (dist_mat_an[i])
        #with: cos(a,p) - 1 < cos(a,n) < cos(a,p)
        #only consider: cos(a,n) < cos(a,p)
        id_an_semi = torch.argmax(dist_mat_an,1) #semi-hardest sample
        print ('semi-hard-neg id_an_semi')
        #print (id_an_semi)
        for i in range(N):
            #print ('semi-hardest value:')
            #print (dist_mat_an[i][id_an_semi[i]])
            if dist_mat_an[i][id_an_semi[i]] <= -1: #important. if has been changed, use easyest sample.
                print ('need recorrect:'+ str(i))
                #print (dist_mat_an[i][id_an_semi[i]])
                #print ('change dist_mat_an semi['+str(i)+']: ' + str(id_an_semi[i]) + ' to: ' + str(id_an[i]) )
                id_an_semi[i] = id_an[i] #easyest sample

        print ('semi-hard-neg id_an_semi recorrect with easyest sample')
        #print (id_an_semi)
        
        #return id_ap, id_an_semi
        return id_ap, id_an

    @staticmethod
    def normalize(x, axis=1):
        x = 1.0*x / (torch.norm(x, 2, axis, keepdim=True) + 1e-12)
        return x

    @staticmethod
    def cosine_dist(x, y):
        print ('compute cosine:')
        print (x)
        #print (y)
        dist = torch.mm(x,y.t()) #cosine distance because x,y are normalized.
        return dist


def normalize(x, axis=1):
    x = 1.0*x / (torch.norm(x, 2, axis, keepdim=True) + 1e-12)
    return x

def main(args):
    #num_classes = 7985
    num_classes = args.num_classes
    batch_size = args.batch_size
    lr_init = args.lr
    epochs = args.epochs
    margin = args.margin
    G_input_size = 64 
    G_hidden_size = 128
    G_projection_size = 64
    
    D1 = Discriminator_speaker(num_classes).cuda()
    #FC_D1 = FullyConnect_speaker(num_classes).cuda()
    D2 = Discriminator_channel().cuda()
    
    G = Generator(G_input_size, G_hidden_size, G_projection_size, batch_size = batch_size).cuda()
    
    #for data unbalanced . train_demo.xls. librispeech 10, xinshen 10.
    #classes_weight = torch.Tensor([0.3027, 0.3027, 0.3027, 0.3027, 0.3027, 0.3027, 0.3027, 0.3027, 0.3027, 0.3027, 0.0550, 0.0075, 0.0189, 0.1211, 0.0378, 0.0108, 0.1514, 0.2018, 0.0075, 0.0144]).cuda()
    #classes_weight = torch.Tensor([0.0193, 0.0176, 0.0192, 0.0218, 0.0179, 0.0177, 0.0200, 0.0176, 0.0183, 0.0186, 0.0172, 0.0178, 0.0156, 0.0179, 0.0165, 0.0181, 0.0189, 0.0171, 0.0183, 0.0192, 0.0190, 0.0179, 0.0177, 0.0177, 0.0199, 0.0171, 0.0171, 0.0199, 0.0183, 0.0207, 0.0191, 0.0192, 0.0183, 0.0176, 0.0190, 0.0192, 0.0179, 0.0189, 0.0189, 0.0183, 0.0194, 0.0210, 0.0191, 0.0190, 0.0197, 0.0192, 0.0185, 0.0192, 0.0186, 0.0176, 0.0200, 0.0193, 0.0181, 0.0190, 0.0187, 0.0194, 0.0191, 0.0210, 0.0183, 0.0178, 0.0190, 0.0183, 0.0180, 0.0178, 0.0176, 0.0181, 0.0176, 0.0181, 0.0172, 0.0189, 0.0181, 0.0189, 0.0188, 0.0178, 0.0184, 0.0179, 0.0176, 0.0190, 0.0176, 0.0177, 0.0174, 0.0181, 0.0192, 0.0185, 0.0183, 0.0185, 0.0186, 0.0198, 0.0189, 0.0183, 0.0192, 0.0183, 0.0195, 0.0178, 0.0175, 0.0183, 0.0193, 0.0197, 0.0173, 0.0170, 0.0181, 0.0183, 0.0186, 0.0176, 0.0193, 0.0187, 0.0191, 0.0189, 0.0183, 0.0179, 0.0179, 0.0179, 0.0176, 0.0171, 0.0171, 0.0180, 0.0180, 0.0171, 0.0175, 0.0178, 0.0175, 0.0173, 0.0195, 0.0172, 0.0174, 0.0187, 0.0180, 0.0178, 0.0174, 0.0173, 0.0181, 0.0171, 0.0173, 0.0204, 0.0187, 0.0183, 0.0189, 0.0181, 0.0177, 0.0173, 0.0173, 0.0173, 0.0176, 0.0176, 0.0181, 0.0179, 0.0172, 0.0177, 0.0177, 0.0178, 0.0178, 0.0182, 0.0181, 0.0179, 0.0176, 0.0182, 0.0173, 0.0173, 0.0179, 0.0183, 0.0174, 0.0181, 0.0177, 0.0176, 0.0183, 0.0202, 0.0193, 0.0183, 0.0170, 0.0181, 0.0181, 0.0168, 0.0170, 0.0185, 0.0177, 0.0175, 0.0176, 0.0183, 0.0177, 0.0174, 0.0187, 0.0178, 0.0181, 0.0180, 0.0176, 0.0167, 0.0179, 0.0172, 0.0163, 0.0183, 0.0168, 0.0178, 0.0181, 0.0185, 0.0177, 0.0176, 0.0181, 0.0171, 0.0177, 0.0181, 0.0193, 0.0175, 0.0175, 0.0179, 0.0178, 0.0183, 0.0176, 0.0183, 0.0182, 0.0184, 0.0161, 0.0180, 0.0176, 0.0176, 0.0174, 0.0181, 0.0193, 0.0179, 0.0181, 0.0183, 0.0189, 0.0200, 0.0188, 0.0181, 0.0174, 0.0178, 0.0171, 0.0170, 0.0176, 0.0176, 0.0183, 0.0181, 0.0171, 0.0171, 0.0167, 0.0171, 0.0183, 0.0178, 0.0178, 0.0176, 0.0193, 0.0183, 0.0183, 0.0176, 0.0185, 0.0181, 0.0177, 0.0193, 0.0203, 0.0183, 0.0179, 0.0171, 0.0181, 0.0191, 0.0157, 0.0172, 0.0176, 0.0176, 0.0177, 0.0179, 0.0165, 0.0171, 0.0169, 0.0170, 0.0173, 0.0168, 0.0191, 0.0173, 0.0196, 0.0176, 0.0189, 0.0179, 0.0177, 0.0174, 0.0174, 0.0184, 0.0180, 0.0158, 0.0173, 0.0174, 0.0168, 0.0170, 0.0180, 0.0173, 0.0181, 0.0187, 0.0178, 0.0173, 0.0171, 0.0174, 0.0178, 0.0171, 0.0170, 0.0170, 0.0173, 0.0192, 0.0198, 0.0170, 0.0183, 0.0167, 0.0181, 0.0177, 0.0185, 0.0189, 0.0264, 0.0166, 0.0172, 0.0173, 0.0158, 0.0181, 0.0176, 0.0179, 0.0177, 0.0170, 0.0173, 0.0174, 0.0168, 0.0177, 0.0180, 0.0183, 0.0171, 0.0171, 0.0174, 0.0177, 0.0183, 0.0175, 0.0177, 0.0166, 0.0167, 0.0171, 0.0173, 0.0176, 0.0185, 0.0177, 0.0168, 0.0172, 0.0173, 0.0198, 0.0178, 0.0166, 0.0176, 0.0168, 0.0176, 0.0180, 0.0185, 0.0173, 0.0205, 0.0179, 0.0178, 0.0179, 0.0176, 0.0183, 0.0180, 0.0177, 0.0178, 0.0173, 0.0186, 0.0171, 0.0177, 0.0178, 0.0174, 0.0179, 0.0174, 0.0188, 0.0184, 0.0190, 0.0173, 0.0178, 0.0177, 0.0181, 0.0171, 0.0178, 0.0170, 0.0171, 0.0178, 0.0178, 0.0165, 0.0173, 0.0163, 0.0169, 0.5555, 0.625, 0.0961, 0.3333, 0.4545, 0.7142, 0.625, 0.7142, 1.0, 0.8333, 0.5, 0.3846, 0.2173, 0.2272, 0.2, 0.5, 0.4166, 0.625, 0.3846, 1.0, 1.0, 0.625, 0.4545, 0.4166, 0.4545, 0.7142, 1.0, 0.5, 0.25, 0.7142, 0.625, 0.5, 0.625, 1.25, 0.2941, 0.3125, 0.4545, 0.3846, 1.0, 1.0, 0.625, 0.5, 0.625, 0.4166, 0.8333, 0.2272, 0.2380, 0.625, 0.25, 0.1785, 0.4166, 0.625, 0.625, 0.625, 0.625, 0.1666, 1.25, 0.7142, 0.625, 0.8333, 0.125, 1.0, 0.3125, 0.2173, 0.625, 0.7142, 0.625, 0.5555, 0.4166, 0.1020, 0.7142, 0.8333, 0.25, 0.625, 0.7142, 0.5555, 0.2083, 0.625, 0.7142, 0.5555, 0.7142, 0.5, 0.8333, 0.25, 0.4545, 0.0568, 0.4166, 0.2941, 0.2083, 0.625, 0.625, 0.5555, 0.625, 0.1063, 0.7142, 0.625, 0.25, 0.7142, 0.2777, 0.4166, 0.5555, 0.0793, 0.1724, 0.625, 0.25, 0.625, 0.4545, 0.4166, 0.4166, 0.5555, 0.5555, 0.7142, 0.8333, 0.3125, 0.2083, 0.4166, 0.3125, 0.3571, 0.4166, 0.625, 0.3333, 0.625, 0.2941, 0.3571, 0.7142, 0.625, 0.8333, 0.5555, 0.0819, 0.2083, 0.4545, 0.3571, 0.625, 0.8333, 0.3125, 0.2777, 0.5555, 0.7142, 0.625, 0.1136, 0.4545, 0.625, 0.7142, 0.2, 0.625, 0.0595, 0.625, 0.5, 0.2380, 0.4166, 0.625, 0.8333, 0.3846, 0.625, 0.3571, 0.625, 0.5555, 0.625, 0.2083, 0.2941, 0.3333, 0.1724, 0.625, 0.8333, 0.7142, 1.25, 0.4166, 0.4166, 0.4545, 0.8333, 0.4166, 0.8333, 1.25, 0.1388, 0.2380, 0.2941, 0.2083, 0.4166, 0.0714, 1.25, 0.3846, 0.2083, 0.2083, 0.625, 0.3846, 0.2083, 0.7142, 0.625, 0.625, 0.3846, 0.2777, 0.4545, 0.625, 0.7142, 0.1923, 0.8333, 0.3571, 0.7142, 0.3571, 0.625, 0.3333, 0.625, 0.625, 0.5555, 0.8333, 0.2941, 0.5, 0.1219, 0.2941, 0.0735, 0.5555, 0.1282, 0.3125, 0.4166, 0.2380, 1.25, 0.4166, 0.2, 0.8333, 0.625, 0.2941, 0.2631, 0.4545, 0.2777, 1.25, 0.8333, 0.625, 0.2631, 0.5555, 0.1562, 0.3571, 0.2173, 0.3333, 0.7142, 0.8333, 1.0, 0.3125, 0.5555, 0.4545, 0.3846, 0.3846, 0.625, 0.3571, 0.625, 0.2272, 0.3846, 0.1562, 0.3571, 0.5, 0.4545, 0.3333, 0.8333, 0.1785, 1.25, 0.4166, 0.5555, 0.625, 0.625, 0.5, 0.4166, 1.25, 0.5, 0.3125, 0.5555, 0.2272, 0.2941, 0.7142, 1.0, 0.625, 0.3571, 0.5555, 0.625, 0.625, 0.625, 0.625, 1.25, 0.8333, 0.2380, 0.8333, 0.3125, 0.625, 1.0, 0.625, 0.1724, 0.5, 0.625, 0.0757, 0.1666, 0.625, 0.625, 0.625, 0.5555, 0.5, 0.0819, 0.8333, 0.2777, 0.4166, 0.8333, 0.5, 0.2941, 0.1851, 0.3125, 0.7142, 0.8333, 0.2941, 0.4166, 0.1562, 0.625, 0.1515, 0.625, 0.7142, 0.5, 0.625, 0.2941, 0.5, 0.3125, 0.2380, 0.1666, 0.2777, 0.2083, 0.7142, 0.625, 0.3125, 0.7142, 0.5, 0.1515, 0.1562, 0.4545, 1.25, 0.1612, 0.8333, 0.8333, 0.3125, 0.2941, 0.1923, 0.625, 0.4545, 0.5555, 0.3571, 0.5555, 0.4545, 0.4166, 0.5555, 0.625, 0.5555, 0.8333, 0.125, 0.4166, 0.625, 0.625, 0.625, 0.4166, 0.2173, 0.5555, 0.3846, 0.5, 0.4545, 0.1562, 0.625, 0.5, 0.7142, 1.0, 0.1724, 0.3333, 1.0, 0.1282, 0.1428, 0.3125, 0.25, 0.4166, 0.25, 0.1315, 0.25, 0.4166, 0.4545, 0.5555, 0.625, 0.3571, 0.8333, 0.625, 0.625, 0.3125, 0.7142, 0.1515, 0.4545, 0.5, 0.3571, 0.1388, 0.25, 0.4166, 1.25, 0.7142, 0.625, 0.0943, 0.5, 0.7142, 0.7142, 0.7142, 0.1388, 0.5, 0.0877, 0.25, 0.625, 0.7142, 0.625, 0.4166, 0.1612, 0.3333, 0.625, 0.3125, 0.8333, 1.0, 0.625, 0.8333, 0.7142, 0.7142, 0.3333, 0.2380, 0.2941, 0.3125, 0.7142, 0.5555, 0.8333, 0.2083, 0.25, 0.1470, 1.25, 0.8333, 0.3571, 0.5, 0.7142, 0.1666, 0.1190, 0.1063, 0.3571, 0.4545, 0.5, 0.3571, 0.0847, 0.4166, 0.4166, 0.2631, 0.8333, 0.7142, 0.7142, 0.25, 0.625, 0.7142, 0.625, 0.5555, 0.5, 0.1562, 0.3125, 0.625, 0.5555, 0.2941, 0.2380, 0.5, 0.3333, 0.1282, 0.4166, 0.625, 0.625, 0.2380, 0.7142, 0.625, 0.0595, 0.1315, 0.2631, 0.3125, 0.625, 0.1724, 0.1785, 0.4166, 0.625, 0.625, 0.2380, 0.1428, 0.3846, 0.7142, 0.2777, 0.1111, 0.1785, 1.25, 0.0833, 1.25, 1.0, 0.5, 0.4166, 0.8333, 0.8333, 0.2380, 0.8333, 0.8333, 0.625, 0.4545, 0.625, 0.8333, 0.3333, 0.4166, 0.3846, 0.1666, 0.1351, 0.625, 0.5, 0.2, 1.0, 0.8333, 0.3571, 0.625, 0.2380, 0.8333, 0.8333, 0.3846, 0.8333, 0.3125, 0.625, 0.2083, 0.2, 0.5555, 0.5555, 0.2941, 0.5, 1.25, 0.2777, 0.2, 0.0980, 0.5, 0.8333, 0.8333, 1.0, 0.3125, 0.8333, 0.3571, 0.5, 0.3125, 0.7142, 0.5, 0.625, 0.1666, 0.7142, 1.25, 0.5555, 0.1351, 0.625, 0.4545, 0.4166, 1.25, 0.625, 0.7142, 0.625, 0.25, 0.3125, 0.1562, 1.0, 1.25, 0.1470, 0.7142, 0.5, 0.7142, 1.0, 0.7142, 0.625, 1.0, 1.0, 0.625, 0.625, 0.5, 0.1562, 0.625, 0.8333, 0.2380, 0.7142, 0.8333, 0.3571, 0.625, 0.5555, 0.4166, 0.625, 0.4545, 1.25, 0.1785, 0.625, 0.25, 1.0, 0.3333, 0.625, 0.3125, 0.625, 0.7142, 0.7142, 0.4166, 0.5, 0.4166, 0.4545, 0.625, 0.3125, 0.3571, 0.5555, 0.2631, 0.5, 0.5555, 0.7142, 0.4166, 0.1351, 0.4166, 0.125, 0.4166, 0.5555, 0.0909, 0.4545, 0.2083, 0.8333, 0.1351, 0.625, 0.625, 0.4545, 0.1351, 0.2272, 0.8333, 0.5555, 0.3333, 0.625, 0.2083, 0.5, 1.0, 0.4545, 0.25, 0.2777, 0.2941, 0.625, 1.0, 0.7142, 0.625, 0.3333, 0.4545, 0.25, 0.2631, 0.4545, 0.8333, 0.4166, 0.3125, 0.3125, 0.5, 0.2941, 0.625, 0.7142, 0.625, 0.8333, 0.1724, 0.3125, 0.7142, 0.25, 0.8333, 0.5, 0.4545, 0.625, 0.2941, 0.3125, 0.3333, 0.2941, 0.8333, 0.7142, 0.7142, 0.5, 0.1851, 0.5555, 0.625, 0.7142, 0.0892, 1.0, 0.1785, 0.5555, 0.2941, 0.7142, 0.4166, 0.2, 0.625, 0.625, 0.2631, 0.3846, 0.5, 0.5, 0.7142, 0.625, 0.4166, 0.3571, 0.7142, 0.5555, 0.4545, 0.1351, 0.3125, 0.5, 0.625, 0.2631, 0.25, 0.25, 0.5, 0.1515, 0.2941, 0.2, 0.5, 0.2380, 0.4545, 0.3333, 0.5555, 0.4545, 0.7142, 0.0781, 0.625, 0.5, 0.0980, 0.1724, 0.8333, 0.8333, 0.4545, 0.4166, 0.4545, 0.3125, 0.8333, 0.3333, 0.7142, 0.4545, 0.625, 0.1315, 0.3846, 0.5555, 0.625, 0.8333, 0.8333, 0.2777, 0.4166, 0.5555, 0.3125, 0.625, 0.5555, 0.7142, 1.0, 0.5, 0.3125, 0.1612, 0.625, 0.5555, 0.4166, 1.25, 0.2777, 0.4545, 0.25, 0.625, 0.4166, 0.3571, 0.1923, 0.5, 0.7142, 1.0, 0.25, 0.8333, 0.4545, 0.4545, 0.3125, 1.0, 0.2941, 0.0609, 0.625, 1.25, 0.2083, 0.3333, 0.625, 0.1785, 0.25, 1.25, 1.25, 0.3125, 0.625, 0.1351, 0.3125, 0.3571, 1.0, 0.1162, 0.3571, 0.1923, 1.0, 0.7142, 0.625, 0.3571, 0.4545, 0.2083, 0.7142, 0.4545, 0.2777, 1.25, 0.4166, 0.4166, 0.2631, 0.8333, 0.7142, 0.7142, 0.4545, 0.3846, 0.3571, 0.2, 0.2083, 0.2631, 0.625, 0.4545, 0.625, 0.3125, 0.4166, 0.8333, 0.625, 0.7142, 0.3571, 0.1041, 0.4545, 0.2941, 0.2631, 0.4545, 0.7142, 0.4166, 0.625, 0.3846, 1.25, 0.5, 0.5, 0.625, 0.4545, 0.5, 0.5, 0.7142, 0.8333, 0.8333, 0.8333, 1.0, 0.5, 0.625, 0.7142, 0.3125, 0.625, 0.3571, 0.4166, 0.3125, 0.5, 0.625, 0.2631, 0.8333, 0.5555, 0.5, 0.1612, 0.7142, 0.8333, 0.625, 0.1923, 0.3846, 0.1724, 0.25, 0.625, 0.2083, 0.7142, 0.4166, 1.0, 0.625, 0.4545, 0.0714, 0.625, 1.0, 0.3125, 0.7142, 0.625, 0.4166, 0.625, 0.5555, 0.0847, 0.8333, 1.0, 0.2631, 1.0, 0.625, 0.3333, 1.0, 0.8333, 0.2083, 0.5555, 0.7142, 0.1063, 0.2380, 0.4166, 0.4166, 0.8333, 0.2272, 0.4166, 0.4545, 0.625, 0.5, 0.625, 0.2083, 0.3125, 0.1515, 0.5, 0.5555, 1.0, 0.3571, 0.2380, 0.8333, 0.8333, 0.4545, 0.8333, 0.2777, 0.3333, 0.3571, 0.625, 0.7142, 0.5, 0.625, 0.625, 0.5555, 0.625, 0.4166, 0.625, 0.3571, 0.7142, 0.1612, 0.125, 0.625, 0.8333, 0.625, 0.7142, 0.2941, 0.625, 0.625, 0.4166, 0.2380, 0.7142, 0.3125, 0.1162, 0.625, 0.2631, 0.4545, 0.625, 0.7142, 0.8333, 0.2941, 0.1666, 0.3333, 0.1282, 0.625, 0.5, 0.625, 0.4166, 0.7142, 0.625, 0.2, 0.5555, 0.4166, 0.5, 0.8333, 0.625, 0.2173, 0.625, 0.8333, 0.625, 0.1851, 0.625, 0.7142, 0.0595, 0.5, 0.8333, 0.8333, 0.3125, 1.25, 0.625, 0.625, 0.1162, 0.5, 0.625, 0.8333, 0.3846, 0.8333, 0.0549, 0.8333, 0.3846, 0.4166, 0.5, 0.625, 0.7142, 0.5, 0.7142, 0.2941, 0.5, 0.3333, 0.4166, 0.3333, 0.3571, 0.5555, 0.3846, 0.1562, 0.5555, 0.1612, 0.5, 0.5, 0.1785, 0.4166, 0.5555, 0.4545, 0.3125, 0.4166, 0.4166, 1.0, 0.1923, 0.625, 0.3846, 1.0, 0.25, 0.7142, 0.625, 0.625, 0.4545, 0.3846, 0.2777, 0.1785, 0.7142, 0.625, 1.0, 0.3333, 0.8333, 0.1219, 0.5, 1.25, 0.3125, 0.2631, 0.3846, 0.3333, 0.8333, 0.4545, 0.5555, 0.4166, 0.625, 0.625, 0.2941, 0.2083, 0.2, 0.0657, 0.0602, 0.3846, 0.1086, 0.1785, 0.7142, 0.5555, 0.8333, 0.8333, 0.0724, 0.625, 1.25, 0.4166, 0.1785, 0.5555, 0.1136, 0.4166, 0.7142, 0.625, 0.4545, 0.5, 0.7142, 0.8333, 0.2631, 0.5555, 0.8333, 0.625, 0.8333, 0.3846, 0.7142, 0.625, 0.1724, 0.3333, 0.5555, 0.3125, 0.625, 0.625, 0.1724, 0.25, 0.7142, 0.2941, 0.1923, 0.8333, 0.1562, 0.8333, 1.25, 0.625, 0.5555, 0.3333, 0.5, 0.625, 0.2083, 0.4166, 0.625, 0.3571, 0.1851, 1.0, 1.0, 0.8333, 0.4166, 0.625, 0.5555, 0.625, 0.5, 0.3846, 0.3846, 0.3125, 0.5555, 0.8333, 0.1136, 0.7142, 1.25, 0.8333, 0.7142, 0.1724, 0.5555, 0.8333, 0.2380, 0.8333, 0.3571, 0.2083, 0.8333, 0.625, 0.4166, 0.1388, 0.4545, 1.25, 0.5, 0.1923, 0.625, 1.0, 0.8333, 0.3333, 0.8333, 0.625, 0.2941, 0.625, 1.0, 0.5, 0.7142, 1.0, 0.4166, 0.8333, 0.8333, 0.7142, 0.3571, 0.25, 0.3125, 0.25, 0.4166, 0.7142, 0.2083, 0.4166, 0.4545, 0.625, 0.3846, 0.7142, 1.25, 0.4545, 0.25, 0.3125, 0.625, 0.5555, 0.3333, 1.25, 0.5, 0.4166, 0.7142, 0.5, 0.625, 0.8333, 0.4545, 0.2941, 0.3846, 0.8333, 0.1724, 0.625, 1.0, 0.2941, 0.5555, 0.2631, 0.3846, 0.5555, 0.4545, 0.4166, 0.5555, 0.1020, 0.2777, 0.625, 0.625, 0.625, 0.4545, 0.1851, 0.1470, 0.4166, 1.0, 0.7142, 0.4545, 0.2631, 0.3846, 0.8333, 0.625, 0.4545, 1.25, 0.8333, 0.3125, 0.7142, 0.1612, 0.4545, 0.5555, 0.3846, 0.625, 0.625, 0.4545, 0.7142, 0.3125, 0.8333, 0.3846, 0.7142, 0.3333, 0.3571, 0.2631, 0.625, 0.625, 0.1785, 1.0, 0.3846, 0.625, 0.4166, 0.8333, 0.7142, 0.2380, 1.25, 0.5555, 0.3125, 1.0, 0.7142, 0.7142, 0.1351, 0.625, 0.625, 0.625, 0.7142, 0.0892, 0.3125, 0.8333, 0.25, 0.5, 0.3846, 0.1063, 0.3125, 0.1562, 0.625, 0.8333, 0.625, 0.4545, 1.0, 0.625, 0.7142, 0.3571, 1.0, 0.5555, 0.625, 0.7142, 0.4166, 0.3125, 0.7142, 0.625, 0.625, 0.2173, 0.2941, 0.5555, 0.1785, 1.0, 0.1923, 0.3571, 0.7142, 0.3846, 0.625, 0.2083, 0.7142, 0.1923, 0.8333, 0.5, 0.2083, 1.0, 0.2941, 0.8333, 0.4166, 0.7142, 0.7142, 0.5555, 0.4166, 0.3125, 0.3333, 0.4545, 0.1562, 1.25, 0.625, 0.4166, 0.5555, 0.4166, 0.8333, 0.2777, 0.7142, 0.625, 0.25, 0.3125, 0.5, 1.0, 0.2272, 0.7142, 0.4166, 0.3846, 0.7142, 0.7142, 0.625, 0.4545, 0.4166, 0.2777, 0.4545, 0.625, 0.4166, 0.8333, 0.4545, 0.25, 0.7142, 0.3846, 0.4545, 0.2380, 0.3125, 0.7142, 0.7142, 0.8333, 0.625, 0.1785, 0.8333, 0.4545, 0.3571, 0.2941, 0.7142, 0.7142, 0.3846, 0.8333, 0.5, 0.5, 0.3125, 0.4545, 0.8333, 0.4166, 0.1388, 0.8333, 0.2777, 0.3571, 0.2272, 0.625, 0.2380, 1.0, 0.8333, 0.1923, 0.0961, 1.0, 1.0, 0.8333, 0.5555, 0.8333, 0.7142, 0.4166, 1.25, 0.8333, 0.2272, 0.4545, 0.625, 1.0, 0.7142, 0.2941, 0.2, 0.3571, 0.3125, 0.25, 0.5555, 0.4545, 0.5, 0.2631, 0.7142, 0.4166, 0.625, 0.625, 0.625, 0.7142, 0.3125, 0.2631, 0.8333, 0.125, 0.1351, 0.4166, 0.4166, 0.4166, 0.4545, 0.3125, 0.3571, 0.4166, 0.3846, 0.625, 0.3571, 0.625, 0.625, 0.4545, 0.1388, 0.2631, 0.2272, 0.2173, 0.7142, 0.5, 0.7142, 0.625, 0.2, 0.5, 0.7142, 0.3125, 0.8333, 0.5555, 0.625, 0.8333, 0.7142, 0.1136, 1.0, 0.2083, 0.8333, 0.4545, 0.625, 0.7142, 0.625, 0.4166, 1.0, 0.7142, 0.7142, 0.5, 1.25, 0.3125, 0.625, 1.25, 0.4166, 0.7142, 0.2, 0.625, 0.2941, 0.2631, 0.1666, 0.625, 0.625, 0.7142, 0.8333, 0.4166, 0.5555, 0.8333, 0.125, 0.8333, 0.3571, 0.5, 0.4545, 0.625, 0.3333, 0.1923, 0.625, 0.2083, 0.3846, 0.625, 0.625, 0.625, 0.8333, 0.2941, 0.8333, 0.3125, 0.7142, 0.3571, 0.625, 0.1724, 0.7142, 0.5555, 0.3125, 0.625, 0.8333, 0.3125, 0.1785, 0.5555, 0.5, 0.8333, 1.0, 0.625, 0.625, 0.0431, 0.4545, 0.2083, 0.1923, 0.5, 0.1785, 0.8333, 0.2173, 0.7142, 0.7142, 0.8333, 0.8333, 0.8333, 0.5555, 0.625, 1.0, 0.3333, 0.1785, 1.25, 0.4166, 1.0, 1.0, 0.8333, 0.1923, 0.4166, 0.1351, 0.5555, 0.625, 0.2777, 0.5555, 0.2, 0.8333, 0.8333, 1.25, 0.625, 0.2272, 1.0, 0.25, 0.625, 0.1020, 1.0, 0.3125, 0.1020, 1.0, 0.7142, 0.0847, 0.5, 1.0, 0.5, 0.625, 0.625, 0.625, 0.4166, 0.7142, 0.8333, 0.2777, 0.3333, 0.625, 0.5555, 0.8333, 0.125, 0.7142, 0.625, 0.3125, 0.5, 0.4166, 0.5, 0.8333, 0.5, 0.7142, 0.8333, 0.2083, 0.4545, 0.1612, 0.8333, 0.625, 1.0, 0.5555, 0.1388, 0.625, 0.0735, 0.3846, 0.625, 0.625, 0.3125, 0.625, 0.1086, 0.1515, 0.3125, 0.2777, 0.5, 0.625, 0.3846, 0.2941, 0.3846, 0.7142, 0.4545, 0.1666, 0.0367, 0.0490, 0.3333, 0.2380, 0.625, 0.3125, 0.5, 1.0, 0.25, 1.0, 0.7142, 0.3125, 1.0, 1.0, 0.8333, 0.8333, 0.5555, 0.625, 1.0, 1.0, 0.4166, 0.1562, 0.2941, 0.4166, 0.7142, 0.8333, 0.2272, 0.2941, 0.2631, 0.2380, 0.625, 0.0925, 0.4166, 0.7142, 0.2941, 0.4545, 0.5555, 0.7142, 0.8333, 0.625, 0.3846, 0.8333, 0.3571, 0.3125, 0.5555, 0.625, 0.1785, 0.2777, 0.0961, 0.625, 0.7142, 0.5, 0.0390, 0.8333, 0.7142, 0.125, 1.0, 0.4166, 0.5555, 0.4166, 0.625, 0.8333, 0.4545, 0.4545, 0.8333, 0.4166, 0.3571, 0.7142, 0.0568, 0.3571, 0.3333, 0.7142, 0.2777, 0.3333, 0.8333, 0.4166, 0.4545, 0.8333, 0.5555, 0.4545, 0.7142, 0.7142, 0.4166, 0.7142, 0.3125, 0.625, 0.3571, 0.5, 0.4166, 0.5, 0.3846, 0.7142, 0.625, 0.2941, 0.8333, 0.3125, 0.4545, 0.7142, 0.25, 0.8333, 0.2941, 0.2941, 0.5555, 0.8333, 0.7142, 0.0793, 0.5, 0.1923, 0.3846, 0.4545, 1.0, 0.7142, 0.5, 0.7142, 0.3333, 0.2631, 0.625, 0.5, 0.3571, 1.0, 1.25, 1.25, 0.1470, 0.1282, 1.25, 0.4166, 0.8333, 0.2272, 0.8333, 0.5555, 0.8333, 0.5, 0.8333, 0.7142, 0.25, 0.3846, 0.125, 0.7142, 0.5, 0.625, 0.4166, 1.25, 0.5555, 0.5555, 0.625, 0.8333, 0.4545, 0.2777, 0.3571, 0.7142, 0.4166, 0.1666, 0.625, 0.1724, 0.7142, 0.4166, 0.3125, 0.3846, 0.4545, 0.5, 0.4166, 0.3846, 0.3571, 0.25, 0.625, 0.5, 0.7142, 0.1851, 1.0, 0.625, 0.4545, 0.625, 0.3125, 0.7142, 0.4545, 0.5, 0.5, 0.8333, 0.2777, 0.25, 0.625, 0.3333, 0.4545, 0.2380, 0.5, 0.8333, 0.8333, 0.0549, 0.8333, 0.1785, 0.3333, 1.0, 0.2941, 0.625, 0.2, 1.25, 0.625, 0.625, 0.5, 1.0, 0.625, 0.1666, 0.4166, 0.4166, 0.8333, 0.5, 0.3846, 0.5555, 0.625, 0.4545, 0.7142, 0.4545, 0.2173, 0.4166, 0.8333, 0.5, 1.0, 0.7142, 0.3125, 0.1923, 0.4545, 0.4166, 0.4166, 0.2083, 0.625, 0.8333, 0.5555, 0.3333, 0.3125, 0.3333, 0.625, 0.7142, 0.4166, 0.5, 0.4545, 0.1785, 0.8333, 0.7142, 0.0847, 0.7142, 1.25, 0.625, 0.625, 1.0, 0.625, 0.2083, 0.4166, 0.5555, 0.7142, 0.4166, 0.1111, 0.25, 0.5, 0.2, 0.5555, 0.625, 0.25, 0.3125, 0.4545, 0.7142, 0.625, 0.3846, 0.7142, 0.7142, 0.4166, 0.5, 0.2380, 0.0657, 0.625, 0.4166, 0.625, 0.3571, 0.5555, 0.1162, 0.2173, 0.4166, 0.7142, 0.8333, 0.5555, 1.25, 1.0, 0.5555, 0.5, 0.7142, 0.3571, 1.25, 0.625, 0.7142, 0.625, 0.3333, 0.625, 0.2941, 0.2173, 0.25, 0.3571, 0.7142, 0.5555, 0.8333, 0.8333, 0.3571, 0.1562, 0.4166, 0.5, 0.7142, 0.7142, 0.625, 0.3333, 0.8333, 0.3333, 0.5555, 0.625, 0.5555, 0.4545, 0.8333, 0.8333, 1.0, 1.25, 1.0, 0.4545, 1.25, 0.2631, 1.25, 1.25, 1.0, 0.4545, 0.8333, 1.0, 1.25, 0.625, 0.625, 0.625, 1.25, 1.0, 1.25, 1.0, 0.8333, 0.625, 0.625, 0.8333, 0.4166, 1.0, 0.4166, 0.1785, 0.7142, 0.7142, 0.4545, 0.1282, 0.625, 0.2272, 0.1785, 0.8333, 0.8333, 0.625, 0.625, 0.3571, 0.3125, 0.25, 0.3333, 0.4166, 1.25, 0.1923, 0.0943, 0.2941, 0.4166, 0.4166, 0.3333, 0.3125, 0.625, 0.3333, 0.4166, 0.8333, 0.2, 0.3125, 0.8333, 0.4166, 0.7142, 0.8333, 0.5, 0.4166, 0.25, 0.625, 0.625, 0.625, 0.4166, 0.625, 0.3333, 0.3846, 0.625, 0.1219, 0.2777, 0.2272, 0.2272, 0.2777, 0.625, 0.5555, 0.625, 0.5, 0.2380, 0.3333, 0.5, 0.0588, 0.3846, 0.3125, 0.8333, 0.4166, 0.5, 0.1923, 0.4166, 0.3846, 0.3846, 0.625, 0.1562, 0.625, 0.5555, 0.625, 0.8333, 0.1785, 0.625, 0.5555, 0.8333, 0.2083, 0.4166, 0.625, 0.25, 0.3333, 0.4166, 0.25, 0.1785, 0.4545, 0.7142, 0.0961, 0.7142, 0.5, 0.8333, 0.5, 0.5555, 0.3333, 0.2631, 0.3125, 1.0, 0.5, 0.7142, 1.0, 0.625, 1.0, 0.4545, 0.5555, 0.2941, 0.4545, 0.125, 0.3571, 0.5555, 0.5555, 0.7142, 1.25, 0.4166, 0.4545, 0.5555, 0.8333, 0.7142, 0.2631, 0.3125, 1.0, 0.4545, 0.3125, 0.1785, 0.3125, 0.7142, 0.1562, 0.25, 0.625, 0.7142, 0.5555, 0.3571, 0.1162, 0.3846, 0.625, 0.625, 0.1923, 1.25, 0.4166, 1.0, 0.5555, 0.5555, 0.5555, 0.625, 0.2083, 1.25, 0.2380, 0.7142, 0.4545, 1.25, 0.2777, 1.0, 0.5555, 0.3125, 0.625, 0.2, 0.7142, 0.25, 0.5555, 0.5555, 0.4166, 0.4166, 0.2173, 0.5, 0.3846, 0.2380, 0.3125, 0.1851, 0.625, 1.0, 0.4166, 0.625, 0.1785, 0.7142, 0.3333, 0.1351, 0.3846, 0.4545, 0.3333, 0.4166, 0.7142, 0.625, 0.3846, 0.8333, 1.0, 1.0, 1.25, 0.3571, 0.8333, 0.8333, 0.2083, 0.2083, 0.5, 0.5555, 0.625, 0.4166, 0.2173, 0.5555, 1.0, 0.1666, 0.625, 0.1388, 0.3125, 0.1785, 0.8333, 0.5, 0.5, 0.3333, 0.4166, 0.625, 0.1785, 0.3846, 0.4545, 0.8333, 0.2380, 0.1785, 0.5555, 0.8333, 0.8333, 0.625, 0.4545, 0.8333, 0.2631, 0.1470, 0.25, 0.4166, 0.7142, 0.3333, 0.2631, 0.5, 0.3571, 0.625, 0.625, 0.1562, 0.3125, 0.5, 0.1162, 0.7142, 0.5555, 0.3333, 0.625, 0.7142, 0.3571, 0.3125, 0.625, 1.0, 0.5, 0.3333, 0.7142, 0.4166, 0.8333, 0.2941, 0.2777, 0.5555, 0.8333, 0.1282, 1.0, 0.625, 0.4166, 0.4166, 1.25, 0.8333, 0.7142, 0.5555, 0.5555, 0.2941, 0.3333, 0.625, 1.25, 1.0, 1.0, 0.8333, 0.4166, 0.625, 0.5, 0.5555, 0.8333, 0.5, 0.2631, 0.5555, 0.2380, 0.3333, 0.3125, 0.8333, 1.25, 0.5, 0.4166, 0.5555, 1.0, 0.1470, 1.0, 0.3846, 0.3571, 0.5, 0.0625, 0.625, 0.7142, 0.7142, 0.8333, 0.5, 0.4166, 0.4545, 1.0, 0.4166, 0.5555, 1.0, 0.4166, 0.5, 0.3571, 0.5555, 0.1785, 0.8333, 0.4166, 0.7142, 0.4545, 0.5, 0.7142, 0.7142, 0.5, 0.625, 0.4166, 0.625, 0.0961, 0.5555, 0.5555, 0.1136, 0.8333, 0.7142, 0.625, 0.625, 0.3846, 0.125, 0.7142, 0.7142, 0.7142, 0.625, 0.7142, 0.4545, 0.2777, 0.25, 0.625, 0.8333, 0.2777, 0.7142, 0.3846, 0.3846, 0.4545, 0.4545, 0.8333, 0.3846, 0.625, 0.5555, 0.625, 0.25, 0.2083, 0.3333, 0.7142, 0.0925, 0.3571, 1.25, 0.7142, 0.625, 0.3125, 1.0, 0.5555, 1.0, 0.8333, 0.0304, 0.8333, 0.2380, 0.4166, 1.25, 0.625, 0.7142, 0.7142, 0.4545, 0.2941, 0.1923, 0.3571, 0.4166, 0.3846, 0.8333, 1.0, 0.7142, 0.3333, 0.2272, 0.2631, 0.5555, 0.3333, 0.625, 0.625, 1.25, 0.3333, 0.4166, 0.4166, 0.0862, 0.2777, 0.3125, 0.625, 0.2941, 1.0, 0.7142, 0.0892, 0.625, 0.4545, 0.2, 0.1785, 0.4166, 0.625, 1.0, 0.4166, 0.2941, 1.0, 0.8333, 1.0, 0.25, 0.7142, 0.0961, 0.4545, 0.3125, 0.3846, 0.625, 0.8333, 0.5, 0.4166, 0.8333, 0.1515, 0.7142, 0.4166, 0.2083, 0.125, 0.4545, 0.2777, 1.25, 0.2272, 0.8333, 0.7142, 0.8333, 0.25, 1.0, 0.2173, 0.2380, 0.625, 1.0, 0.4166, 0.1562, 0.125, 0.25, 0.4545, 0.3571, 1.25, 0.625, 0.4545, 0.5, 0.4545, 0.4166, 0.5555, 1.0, 0.8333, 0.8333, 0.25, 0.7142, 1.0, 0.4166, 0.7142, 0.7142, 0.625, 0.3571, 0.3125, 0.4166, 1.25, 0.625, 0.5, 0.2083, 1.0, 0.4166, 0.7142, 0.4166, 0.8333, 0.3125, 0.625, 0.8333, 0.25, 0.7142, 0.5555, 0.2272, 0.3125, 0.1785, 0.3125, 0.4545, 0.0704, 0.8333, 0.4545, 0.3125, 0.5555, 0.625, 0.5555, 0.7142, 0.4545, 0.625, 0.4166, 0.1515, 0.625, 0.625, 1.0, 0.625, 0.4545, 0.25, 1.0, 0.4166, 0.25, 0.5, 1.25, 0.7142, 0.7142, 0.4545, 1.0, 0.0357, 0.3125, 0.3846, 0.2083, 0.4166, 1.25, 0.625, 0.25, 0.4166, 0.5, 1.25, 0.1136, 0.625, 0.625, 0.5555, 0.2380, 0.3846, 0.7142, 0.7142, 0.8333, 0.2380, 0.7142, 0.3333, 0.625, 0.1851, 0.1190, 0.8333, 0.0485, 0.625, 0.5555, 0.625, 0.1351, 0.3333, 0.5555, 0.4166, 0.4166, 0.7142, 0.5555, 0.5, 0.2, 0.8333, 0.5, 0.8333, 0.5, 0.4166, 0.7142, 0.8333, 0.1923, 0.7142, 0.3846, 0.1923, 0.3333, 0.0675, 0.1612, 0.7142, 0.2941, 0.3333, 0.625, 0.2173, 0.4545, 0.4166, 0.625, 0.3125, 0.4545, 0.2380, 0.4545, 0.8333, 0.1851, 0.2777, 0.625, 0.4545, 0.3125, 0.4166, 0.3125, 0.5, 0.1515, 0.4166, 0.1041, 0.1136, 0.625, 0.2941, 0.625, 0.5555, 0.8333, 0.625, 0.4166, 0.5555, 0.8333, 0.4166, 0.4166, 0.5, 0.7142, 0.625, 0.25, 0.4166, 0.3846, 0.5555, 0.8333, 0.625, 0.4166, 0.8333, 0.4545, 0.7142, 1.0, 0.5555, 0.7142, 0.3125, 0.1562, 0.8333, 1.25, 0.1428, 1.0, 0.2380, 0.7142, 0.4166, 0.3571, 1.0, 0.3846, 0.8333, 0.1785, 0.7142, 0.625, 0.0746, 0.5, 0.1612, 0.3846, 0.7142, 0.3846, 0.1351, 0.8333, 0.3571, 1.0, 0.1190, 0.2941, 0.2777, 0.2272, 0.4166, 0.0892, 1.25, 0.4166, 0.8333, 0.3571, 0.5555, 0.3125, 0.625, 0.7142, 0.4166, 1.25, 0.4545, 0.3125, 0.4545, 0.625, 0.625, 0.4166, 0.3846, 0.4166, 0.2941, 0.1388, 0.8333, 0.3571, 0.125, 0.5, 0.8333, 0.1785, 0.7142, 0.7142, 0.8333, 0.2380, 0.8333, 0.4545, 0.7142, 0.625, 0.2083, 0.8333, 0.625, 0.5, 0.5, 0.3571, 0.8333, 0.8333, 0.625, 0.625, 0.4166, 0.3571, 1.0, 0.7142, 0.7142, 0.5, 0.2777, 0.4166, 0.1785, 0.7142, 0.625, 0.4166, 0.3125, 0.625, 0.8333, 0.5, 0.3846, 0.3333, 0.3125, 0.5555, 0.5, 0.5, 0.625, 0.3333, 0.2083, 0.625, 0.4166, 0.5, 0.1282, 0.625, 0.2941, 0.3846, 0.4545, 0.1351, 0.625, 0.625, 0.7142, 0.3333, 0.5555, 1.25, 0.625, 0.1562, 0.5555, 0.5, 0.4166, 0.5, 0.2777, 0.7142, 0.8333, 0.2083, 1.0, 0.625, 0.2941, 0.5555, 0.7142, 0.625, 0.625, 0.2380, 0.2083, 0.625, 0.625, 0.2777, 0.2380, 0.4545, 0.625, 0.5555, 1.0, 0.4166, 0.2631, 1.25, 0.7142, 0.625, 0.3125, 0.1388, 0.4166, 0.3333, 0.4166, 0.25, 0.3333, 0.4166, 0.3125, 0.3571, 0.4166, 0.625, 0.625, 0.625, 0.1612, 0.5555, 0.625, 0.625, 0.4545, 0.1562, 0.7142, 0.5, 0.1111, 0.3846, 0.8333, 0.2083, 0.4166, 0.8333, 0.4166, 0.1666, 0.2380, 0.8333, 0.3846, 0.625, 0.3571, 0.625, 0.5, 0.0724, 0.625, 0.5, 0.5555, 1.0, 0.1785, 0.5, 0.8333, 0.5, 0.4545, 0.4166, 0.625, 0.625, 0.5, 0.625, 0.3571, 0.7142, 0.1785, 0.625, 0.25, 1.0, 0.7142, 0.7142, 0.3333, 0.625, 0.2777, 0.4545, 0.625, 0.7142, 0.5555, 0.2380, 0.625, 0.3125, 0.8333, 0.3125, 0.5, 0.4545, 0.5, 0.3571, 0.3846, 0.3846, 0.625, 0.5555, 0.625, 0.4166, 0.4545, 0.625, 0.5, 0.7142, 0.5555, 0.625, 0.2777, 0.1923, 0.4545, 0.625, 0.5, 0.5, 0.625, 0.1428, 0.1923, 0.4166, 0.5555, 0.25, 0.3125, 0.625, 0.25, 0.3125, 0.5555, 0.5555, 0.2777, 0.5, 0.5, 0.8333, 0.1162, 0.4545, 0.3125, 0.8333, 0.3846, 0.4545, 0.5555, 0.1388, 0.8333, 0.25, 0.1111, 0.625, 0.1724, 0.7142, 0.625, 0.3846, 0.4166, 0.4166, 0.1, 0.4166, 0.2631, 0.8333, 0.625, 0.8333, 0.625, 0.5555, 0.3571, 0.5, 0.7142, 0.625, 0.7142, 0.3846, 0.7142, 0.5555, 0.0781, 0.7142, 0.625, 0.3125, 0.7142, 0.3846, 0.8333, 0.4545, 0.2631, 0.7142, 0.8333, 0.7142, 0.4545, 0.3333, 1.0, 1.25, 0.5, 0.2631, 0.3846, 0.625, 0.2083, 0.7142, 0.7142, 0.5555, 0.8333, 0.4545, 0.4166, 0.7142, 0.25, 1.0, 0.4166, 0.3571, 0.4166, 0.4166, 0.625, 1.0, 0.2173, 0.5, 0.5555, 0.5, 0.625, 0.2777, 0.2, 0.5555, 0.625, 0.5555, 0.625, 0.625, 0.4166, 0.625, 0.5, 0.4166, 0.8333, 0.1388, 1.25, 0.4166, 1.25, 0.625, 0.625, 1.0, 0.7142, 0.8333, 0.1851, 0.5, 0.2941, 0.625, 0.4166, 0.5555, 0.4166, 0.2173, 0.5555, 0.1162, 0.7142, 0.3333, 0.8333, 0.1562, 0.7142, 0.4166, 0.3125, 0.3846, 1.25, 0.625, 0.3125, 0.1785, 0.625, 0.4166, 0.7142, 0.3333, 0.2941, 0.3846, 0.2272, 0.3571, 1.0, 0.8333, 0.625, 0.2777, 0.05, 0.8333, 0.5555, 0.5, 0.2173, 0.25, 0.8333, 0.4545, 0.8333, 0.3125, 0.1666, 0.2631, 0.4166, 0.625, 0.2083, 0.625, 0.8333, 0.8333, 0.4166, 0.5, 0.8333, 0.625, 0.3333, 0.2777, 0.4166, 0.3333, 0.7142, 0.3125, 0.2380, 0.625, 0.3333, 0.0609, 0.7142, 0.625, 0.8333, 0.4166, 0.8333, 0.1428, 0.7142, 0.4545, 0.8333, 0.4545, 0.4545, 0.0704, 0.625, 0.2631, 0.8333, 0.4545, 0.8333, 0.1315, 0.2777, 0.4545, 0.625, 0.3571, 0.4545, 0.4166, 0.7142, 0.5555, 0.3125, 1.0, 0.4166, 0.4166, 0.5, 0.4545, 0.5, 1.0, 0.2777, 0.4166, 0.625, 0.1785, 0.8333, 0.8333, 0.4545, 0.4166, 0.5, 0.625, 0.1428, 0.625, 0.25, 1.0, 0.7142, 0.3125, 0.2941, 0.625, 0.7142, 0.625, 0.0943, 0.7142, 0.7142, 0.625, 0.5555, 0.7142, 0.2083, 0.1136, 0.2941, 0.7142, 0.3333, 0.625, 0.8333, 0.625, 0.7142, 0.4166, 0.625, 0.4166, 1.0, 0.5555, 0.5, 0.5, 0.4545, 0.5555, 1.25, 0.5555, 0.1086, 0.5, 0.2631, 0.1136, 0.0434, 0.5555, 0.2173, 0.2380, 0.8333, 0.3125, 0.4166, 0.3333, 0.7142, 1.0, 0.625, 0.3333, 0.4166, 0.5, 0.2631, 1.0, 0.3846, 0.4545, 0.625, 0.8333, 0.4545, 0.8333, 0.625, 0.25, 0.4545, 0.1785, 1.0, 0.3125, 0.2941, 0.625, 0.25, 0.3846, 0.2, 0.4545, 0.625, 0.3333, 0.1562, 0.3571, 0.4545, 1.25, 0.1851, 0.8333, 0.7142, 0.625, 0.625, 0.5, 1.0, 0.5555, 0.8333, 0.5, 0.4545, 0.625, 0.25, 0.5, 0.1428, 1.0, 1.25, 0.625, 1.0, 0.1515, 0.4166, 0.5, 1.0, 0.1086, 0.125, 0.4545, 0.3125, 0.8333, 0.3846, 0.8333, 0.2380, 0.3125, 0.0704, 0.625, 0.8333, 0.25, 0.2777, 0.25, 0.625, 0.3846, 0.5, 0.1063, 0.625, 0.2, 0.8333, 0.5555, 0.2173, 0.625, 0.3571, 0.4166, 0.2631, 0.625, 0.7142, 0.4166, 0.2631, 0.7142, 0.625, 1.25, 0.1923, 0.625, 0.1851, 0.1724, 0.4545, 0.1190, 0.1785, 0.3846, 0.1923, 0.7142, 0.7142, 0.3333, 0.3333, 0.3333, 0.8333, 0.7142, 0.3846, 0.625, 0.8333, 1.25, 1.0, 0.1612, 0.2083, 0.125, 0.25, 0.7142, 0.5, 0.7142, 0.2173, 0.7142, 0.625, 0.625, 0.2631, 0.5555, 0.7142, 0.2941, 0.3333, 0.8333, 1.0, 0.8333, 0.2083, 0.3125, 0.625, 0.3125, 0.5555, 0.8333, 0.3125, 0.2, 0.625, 0.1923, 0.1785, 0.4166, 0.2173, 0.1612, 0.8333, 0.3125, 0.5555, 0.1136, 0.625, 0.5555, 0.3571, 0.4545, 0.5555, 0.7142, 0.3571, 0.7142, 1.0, 1.0, 0.1724, 0.8333, 0.2083, 0.2, 0.4545, 0.625, 0.2777, 0.7142, 0.3846, 0.1219, 0.5, 0.0704, 0.4166, 0.0568, 0.5, 0.0862, 0.3846, 0.4545, 1.0, 0.5555, 0.625, 0.8333, 0.8333, 0.625, 1.0, 0.1923, 0.5555, 0.625, 0.4545, 0.625, 0.625, 0.8333, 0.7142, 0.4166, 0.5555, 0.2631, 0.625, 0.625, 0.4545, 0.1351, 0.5, 0.625, 0.7142, 0.3846, 0.2272, 0.4545, 0.1785, 0.4166, 0.625, 0.5, 0.5555, 0.7142, 0.5, 0.2083, 0.625, 0.3571, 0.4166, 1.25, 0.625, 0.7142, 0.8333, 0.3333, 0.3846, 0.7142, 0.5555, 0.3846, 0.3846, 1.0, 0.625, 0.4166, 1.25, 0.3571, 0.5, 0.1041, 0.7142, 0.5, 0.625, 0.1, 0.7142, 0.5555, 0.3125, 1.25, 0.2380, 0.3571, 0.1315, 0.2777, 0.625, 0.8333, 1.25, 1.25, 0.5, 1.0, 0.7142, 0.625, 0.3125, 0.4166, 0.3571, 0.3125, 0.4166, 0.4545, 0.25, 0.625, 0.625, 0.4166, 0.8333, 0.625, 0.625, 0.25, 0.4166, 0.625, 0.625, 0.4545, 0.4545, 0.5, 0.1785, 0.7142, 0.8333, 0.7142, 0.625, 0.8333, 0.3571, 0.3846, 0.4166, 0.2083, 0.5, 0.8333, 0.25, 1.25, 0.4166, 0.7142, 0.625, 0.7142, 0.7142, 0.1351, 0.3333, 0.3125, 0.1785, 0.4166, 0.5, 0.3125, 0.4166, 0.8333, 0.25, 0.2272, 0.3333, 0.3125, 0.8333, 1.0, 0.625, 0.8333, 0.8333, 0.3125, 0.7142, 0.8333, 0.2631, 0.2631, 0.5, 0.4166, 0.4545, 0.7142, 0.7142, 0.7142, 0.625, 0.2631, 0.625, 0.8333, 0.2631, 0.4166, 0.4545, 0.125, 0.3846, 0.0943, 0.2083, 0.5, 0.2380, 0.625, 0.625, 0.8333, 0.8333, 0.5, 0.625, 1.0, 0.625, 0.4166, 0.625, 0.5555, 1.0, 0.5555, 0.1851, 0.4166, 0.3846, 0.1785, 0.625, 1.25, 0.3125, 0.3333, 0.1923, 0.8333, 0.25, 0.625, 0.625, 0.3333, 0.5555, 0.8333, 0.3846, 0.3846, 0.7142, 0.7142, 0.3846, 0.4166, 1.0, 0.2380, 0.3571, 0.625, 0.1851, 0.625, 0.7142, 0.625, 0.7142, 0.8333, 0.625, 0.2777, 0.625, 0.4166, 0.3125, 0.8333, 0.625, 0.625, 0.3125, 0.3846, 1.0, 0.25, 0.25, 0.4166, 0.625, 0.2777, 0.625, 0.625, 0.625, 0.625, 0.4166, 0.4166, 0.3571, 0.7142, 0.5, 0.125, 0.7142, 0.25, 0.4166, 0.1785, 0.625, 0.2777, 0.625, 0.5555, 0.4166, 0.7142, 0.8333, 0.7142, 0.625, 0.25, 0.5, 0.2380, 0.5555, 0.3571, 0.625, 0.7142, 0.8333, 0.5555, 0.3125, 0.8333, 0.7142, 0.5, 0.4545, 1.0, 0.5, 0.3846, 0.7142, 0.625, 1.0, 0.5555, 0.7142, 1.0, 0.7142, 0.625, 0.8333, 0.25, 0.625, 0.25, 0.25, 0.3846, 0.3333, 0.1785, 0.3846, 0.8333, 0.8333, 0.4545, 0.5, 0.3125, 0.625, 0.625, 0.5555, 1.25, 0.3333, 0.4545, 0.4545, 0.2941, 0.4166, 0.3125, 0.2083, 0.4166, 0.2272, 0.4545, 0.625, 0.3846, 0.625, 0.7142, 0.7142, 0.5555, 0.2631, 0.7142, 0.5, 0.2631, 0.4166, 0.7142, 0.1612, 0.8333, 0.3846, 0.625, 0.4166, 0.5555, 0.5555, 0.2777, 0.8333, 0.4166, 0.2380, 0.3333, 0.8333, 0.4545, 0.2, 0.625, 0.5555, 0.4166, 0.8333, 0.4545, 0.3333, 0.7142, 0.3125, 0.3571, 0.8333, 0.3846, 0.4545, 0.5, 0.2777, 0.7142, 0.25, 0.5, 0.3571, 0.2777, 0.2631, 0.3333, 0.5, 0.2380, 0.7142, 0.25, 0.2, 0.7142, 0.4166, 0.625, 0.2777, 0.625, 0.8333, 0.4545, 0.7142, 0.7142, 0.5, 0.625, 0.4166, 0.8333, 0.2380, 0.3125, 1.0, 0.3333, 0.8333, 0.3846, 0.4166, 0.3846, 0.2941, 0.1724, 0.4166, 0.625, 0.625, 0.625, 0.0381, 0.5555, 1.0, 0.625, 0.5555, 1.0, 0.3333, 0.1162, 0.25, 1.0, 0.3846, 0.8333, 0.3125, 0.4545, 0.2631, 0.4166, 1.0, 0.8333, 0.7142, 0.1612, 0.2083, 0.625, 0.625, 0.5555, 0.3125, 0.625, 0.3571, 1.25, 0.8333, 1.0, 0.7142, 0.2941, 0.4166, 0.4166, 0.7142, 1.0, 0.5, 0.2272, 0.7142, 0.3571, 0.3571, 0.3333, 0.2631, 0.4545, 0.5, 0.2631, 1.25, 0.2083, 0.2941, 0.2272, 0.8333, 0.4545, 0.7142, 0.7142, 0.5, 0.1612, 0.625, 1.25, 0.4166, 0.25, 0.8333, 0.5555, 0.625, 0.3846, 0.625, 1.25, 1.0, 0.625, 0.5, 0.2380, 0.3333, 0.7142, 0.2777, 0.5, 0.4166, 0.7142, 0.8333, 0.3846, 0.625, 1.0, 0.3846, 0.625, 0.2941, 0.625, 0.7142, 1.0, 0.25, 0.4166, 0.3125, 0.4545, 0.4545, 0.2380, 0.625, 0.4166, 0.1851, 0.5, 0.0625, 0.7142, 0.0543, 0.625, 0.4545, 0.625, 0.625, 0.5555, 0.5555, 0.4166, 0.625, 0.1562, 0.7142, 0.625, 0.2272, 0.8333, 0.625, 0.8333, 0.25, 0.5555, 1.25, 1.25, 0.0666, 0.4166, 0.4545, 0.3846, 1.0, 0.5555, 1.25, 0.5, 0.3571, 0.1923, 0.625, 1.0, 0.4545, 0.4545, 0.1470, 0.7142, 0.0295, 0.2777, 0.5, 0.2631, 0.7142, 0.5555, 0.8333, 1.0, 0.8333, 0.5555, 0.625, 0.4545, 0.3571, 0.2777, 0.2941, 0.2173, 0.5, 0.4545, 0.3125, 0.7142, 0.5555, 0.3333, 0.2, 0.8333, 0.7142, 0.5, 0.8333, 1.25, 0.8333, 0.7142, 0.2380, 0.25, 0.4166, 1.0, 0.4166, 0.25, 0.2380, 0.1923, 0.3846, 0.1724, 0.4545, 1.0, 0.8333, 0.625, 0.7142, 0.8333, 0.5555, 0.8333, 0.7142, 0.3846, 0.3125, 0.625, 0.7142, 0.8333, 0.5555, 0.2777, 0.2777, 0.5555, 0.7142, 1.0, 0.5555, 0.7142, 0.2631, 0.4545, 0.3125, 0.5555, 1.0, 1.0, 0.0862, 0.625, 0.7142, 0.3333, 0.5, 0.625, 0.25, 1.25, 1.0, 0.3846, 0.2631, 0.625, 0.7142, 0.0793, 0.8333, 0.5, 0.3333, 0.1351, 0.4545, 0.1785, 0.7142, 0.8333, 0.3125, 1.0, 0.8333, 0.625, 0.5, 0.0961, 0.2631, 0.1388, 0.4166, 0.1666, 0.5555, 0.625, 0.8333, 0.4166, 0.3125, 0.625, 0.25, 0.2083, 0.1282, 1.25, 1.0, 0.0649, 0.7142, 0.625, 0.4166, 0.3571, 1.25, 0.7142, 0.7142, 0.8333, 0.625, 0.1282, 0.5, 0.2, 0.625, 1.25, 0.2941, 0.7142, 0.5555, 0.3571, 0.4545, 0.7142, 0.7142, 0.3333, 0.4166, 0.625, 0.5555, 1.25, 0.25, 0.8333, 1.0, 0.5555, 0.1562, 0.4545, 0.4545, 1.25, 0.2631, 0.625, 0.7142, 1.0, 0.625, 0.4545, 0.3125, 0.2777, 0.5, 0.3333, 0.8333, 0.8333, 0.2272, 0.4545, 1.25, 0.3846, 0.2941, 0.625, 0.1388, 0.7142, 0.4545, 0.25, 0.3846, 0.4166, 0.8333, 0.2777, 0.7142, 0.8333, 0.8333, 0.625, 0.8333, 0.3846, 0.625, 0.3125, 0.4545, 0.1724, 0.3125, 0.1666, 0.625, 0.8333, 0.625, 0.4166, 1.0, 0.625, 0.7142, 0.625, 0.1785, 0.2631, 0.7142, 0.4166, 0.7142, 0.5555, 0.3125, 0.2941, 0.7142, 0.625, 0.3571, 0.8333, 1.0, 0.7142, 0.4545, 0.1785, 0.3571, 0.2173, 0.2941, 0.3571, 0.7142, 0.8333, 0.25, 1.25, 0.5555, 1.0, 1.0, 1.0, 0.8333, 1.0, 1.0, 1.25, 1.25, 1.25, 1.25, 0.7142, 1.0, 0.8333, 1.25, 0.625, 0.5, 1.0, 0.8333, 1.25, 1.25, 1.0, 0.8333, 1.25, 0.4545, 1.0, 1.0, 1.25, 0.8333, 0.625, 0.7142, 0.0892, 0.5, 0.25, 0.7142, 0.4166, 0.1515, 0.0833, 0.4545, 0.625, 1.25, 0.7142, 0.4545, 0.4166, 0.5, 0.2631, 0.625, 0.2631, 0.4545, 0.3333, 1.25, 0.8333, 0.2083, 0.4545, 0.4166, 0.3125, 0.7142, 0.2083, 0.2941, 0.8333, 0.4545, 0.2631, 0.625, 0.625, 0.625, 0.1785, 0.8333, 0.4166, 0.7142, 0.5555, 0.8333, 1.0, 0.3125, 0.7142, 0.3571, 0.1612, 0.4545, 0.8333, 0.8333, 0.3571, 0.4545, 0.8333, 0.625, 0.1612, 0.7142, 0.5, 0.5, 0.7142, 0.8333, 0.8333, 0.625, 0.1515, 0.3125, 0.3571, 0.3571, 0.2941, 0.3846, 0.2777, 0.625, 0.3571, 0.8333, 0.3571, 0.3571, 0.1785, 0.8333, 0.625, 0.1041, 0.7142, 0.8333, 0.8333, 0.3333, 0.4166, 0.4545, 0.625, 0.3333, 0.3333, 0.625, 0.3333, 0.3333, 1.25, 0.2083, 0.3571, 0.125, 0.625, 0.8333, 0.3571, 0.5555, 0.4166, 0.4166, 0.5555, 1.0, 0.8333, 1.0, 0.2941, 0.1, 0.7142, 0.3333, 0.7142, 0.4545, 1.0, 0.2777, 0.3846, 0.3571, 0.1785, 0.3125, 0.4545, 0.625, 0.7142, 1.0, 0.625, 0.7142, 0.2173, 0.1470, 0.4166, 0.2941, 0.5555, 0.2380, 0.625, 0.625, 0.2173, 0.2083, 0.5555, 1.25, 0.4166, 0.2083, 0.2380, 0.1785, 0.3846, 0.8333, 0.5555, 0.5555, 0.25, 0.8333, 0.7142, 0.5, 0.1785, 0.3125, 0.2083, 0.2631, 0.5555, 0.625, 0.625, 0.4545, 0.3333, 0.8333, 0.7142, 0.8333, 0.2631, 0.8333, 0.2, 0.1470, 0.1041, 0.4545, 0.25, 0.5, 0.5555, 0.2272, 0.7142, 0.7142, 0.1562, 0.625, 0.5555, 0.8333, 0.2777, 0.3125, 0.7142, 0.4545, 1.25, 0.5, 0.8333, 0.625, 0.4166, 0.2, 0.7142, 0.8333, 1.0, 0.4166, 0.5, 0.1282, 0.625, 0.2380, 0.4545, 0.625, 0.8333, 0.625, 0.7142, 1.0, 0.1515, 0.625, 0.5555, 0.625, 0.7142, 0.7142, 0.3125, 0.5555, 0.5, 0.1562, 0.4166, 0.4166, 0.3571, 0.5555, 0.2, 0.2631, 0.8333, 0.3125, 0.25, 0.3846, 0.4166, 0.625, 0.4166, 1.0, 0.3333, 0.5, 0.3125, 0.625, 0.1785, 0.5555, 0.625, 0.1428, 0.8333, 0.2083, 0.5555, 1.0, 0.625, 0.1724, 0.8333, 0.625, 0.1388, 0.4166, 0.1785, 0.3333, 0.5, 0.1388, 0.625, 0.2083, 0.3125, 0.8333, 0.625, 0.7142, 0.5, 1.25, 0.8333, 1.25, 0.8333, 0.625, 0.3125, 0.2941, 0.8333, 0.3125, 1.0, 0.8333, 0.3571, 0.1428, 0.7142, 0.3571, 0.1851, 0.2272, 0.4545, 0.3125, 0.2941, 0.625, 0.1351, 0.3571, 0.7142, 0.4545, 0.625, 0.4545, 0.7142, 0.4166, 1.0, 0.7142, 0.1923, 0.5, 0.625, 0.3846, 0.3125, 0.2631, 1.0, 1.0, 0.7142, 0.7142, 0.5, 0.3125, 0.625, 0.2941, 0.0378, 0.3125, 0.1923, 0.5555, 0.4545, 0.5555, 0.3571, 0.5555, 0.0925, 1.0, 0.7142, 0.625, 1.0, 0.8333, 0.2631, 0.7142, 0.0595, 0.2631, 0.25, 0.7142, 0.5555, 0.8333, 0.625, 0.3571, 0.4166, 0.2777, 0.625, 0.1388, 0.625, 0.1562, 0.8333, 1.25, 0.2777, 0.8333, 0.5555, 0.2941, 0.5, 0.2941, 0.1162, 0.8333, 0.25, 0.8333, 0.625, 0.2173, 0.0909, 0.4545, 0.625, 0.5555, 0.4166, 0.7142, 0.8333, 0.4166, 0.8333, 0.4166, 0.7142, 0.625, 0.625, 0.7142, 0.625, 0.625, 0.8333, 0.625, 0.8333, 0.3571, 0.625, 0.4166, 0.4545, 0.4166, 0.625, 1.0, 0.2380, 0.625, 0.5555, 0.8333, 0.5, 0.625, 0.3846, 0.625, 0.7142, 0.2083, 0.8333, 0.2, 0.3846, 0.5555, 0.8333, 0.8333, 0.4166, 0.2173, 0.25, 0.4166, 0.5, 0.3846, 0.2083, 0.625, 1.0, 0.4545, 0.2083, 0.5555, 0.625, 0.7142, 0.2380, 0.2631, 0.125, 0.1851, 0.3333, 0.3333, 0.4166, 0.4545, 0.3846, 0.3125, 0.2777, 0.5, 0.8333, 0.3125, 0.2380, 0.3125, 0.2083, 0.25, 0.5555, 0.2272, 0.625, 0.7142, 1.25, 0.1785, 0.8333, 0.1562, 0.8333, 0.5555, 0.5555, 0.1428, 0.625, 1.0, 0.5, 0.625, 0.7142, 0.5555, 0.3846, 0.4166, 0.8333, 0.4166, 1.0, 0.8333, 1.0, 0.5555, 0.7142, 0.3846, 0.5555, 0.5, 0.5, 1.25, 0.4166, 0.625, 0.7142, 0.8333, 0.5, 0.8333, 0.3846, 0.1666, 0.2941, 0.4166, 0.625, 0.25, 0.4166, 0.2777, 0.8333, 0.5, 0.7142, 0.5555, 0.5555, 1.0, 0.4166, 0.25, 0.8333, 0.25, 0.7142, 0.625, 0.625, 0.2631, 0.2173, 0.4166, 0.7142, 0.0684, 0.7142, 1.0, 0.4166, 0.3571, 0.2173, 0.5555, 0.625, 0.7142, 0.2083, 0.625, 0.5, 0.5555, 0.4166, 0.3846, 0.8333, 0.625, 1.0, 0.625, 0.5, 0.4166, 0.7142, 0.4166, 0.7142, 0.3125, 0.625, 1.0, 0.5, 0.8333, 0.8333, 0.2941, 0.625, 0.8333, 0.3571, 0.3125, 1.0, 0.7142, 0.2631, 0.2777, 0.2272, 0.1470, 1.25, 0.5555, 0.3846, 0.3125, 0.2631, 0.4545, 1.0, 0.3125, 0.25, 0.25, 1.25, 0.7142, 0.2631, 0.8333, 0.4166, 0.8333, 0.3333, 0.5, 0.7142, 0.4545, 0.1190, 0.5, 1.25, 0.8333, 0.3125, 0.3125, 0.2272, 0.4166, 0.3846, 0.4166, 0.2083, 0.625, 0.4545, 0.8333, 0.4545, 0.3571, 0.8333, 0.2272, 0.4545, 0.5, 1.0, 0.7142, 0.0806, 0.5555, 0.1515, 0.7142, 1.0, 0.5555, 0.5, 0.7142, 0.3125, 0.625, 0.2631, 1.25, 0.0675, 0.4545, 0.4166, 0.3333, 0.1851, 0.7142, 0.1851, 0.4166, 0.4545, 0.5, 0.1351, 1.0, 0.5, 0.5, 0.5, 0.625, 0.7142, 0.5555, 0.4166, 0.625, 0.625, 0.8333, 0.5555, 0.2631, 0.625, 0.5555, 0.1923, 0.4166, 0.2380, 0.625, 0.8333, 0.3125, 0.7142, 1.25, 0.5, 0.8333, 0.8333, 0.5, 0.3846, 0.8333, 0.4166, 0.4545, 0.5555, 0.625, 1.25, 0.5, 0.3846, 0.5, 0.625, 0.2777, 0.3571, 0.5555, 0.7142, 0.25, 0.5555, 0.4166, 0.7142, 0.3333, 0.1562, 0.2272, 0.625, 0.8333, 0.5555, 0.1041, 0.0641, 1.0, 0.2777, 0.4166, 0.8333, 0.5555, 0.1612, 0.625, 0.3125, 0.25, 0.25, 0.7142, 0.625, 0.625, 0.0694, 0.4545, 0.8333, 0.2380, 0.5555, 0.4545, 0.4545, 0.4166, 0.4166, 0.1020, 0.8333, 0.2631, 0.5555, 0.7142, 0.8333, 0.2380, 0.625, 0.7142, 0.625, 0.3571, 0.4545, 0.8333, 0.8333, 0.5, 0.8333, 0.5555, 0.625, 0.2777, 0.5555, 0.7142, 0.625, 0.4166, 0.25, 0.1923, 1.25, 1.25, 0.3333, 0.25, 0.625, 0.625, 1.0, 0.4166, 0.4545, 0.3571, 1.0, 0.7142, 0.5, 0.2083, 0.5555, 0.1428, 0.7142, 0.4166, 0.4545, 0.5, 0.4166, 0.5555, 0.8333, 0.3571, 0.4166, 0.7142, 0.3125, 0.4545, 0.625, 0.7142, 0.625, 1.0, 0.0510, 0.625, 0.7142, 0.1562, 0.2272, 0.625, 0.3333, 0.8333, 0.7142, 0.4166, 0.625, 0.625, 1.25, 0.25, 0.4545, 1.25, 0.25, 0.8333, 0.625, 0.4545, 0.7142, 0.2083, 0.5555, 0.7142, 0.3571, 0.4545, 0.5555, 0.1666, 0.3571, 0.2941, 0.2272, 1.0, 0.8333, 0.5555, 0.625, 0.3125, 0.5555, 0.4166, 0.5555, 0.4545, 0.625, 0.4545, 0.625, 0.0980, 0.7142, 0.4166, 0.1388, 1.0, 0.5555, 0.8333, 0.1923, 0.8333, 0.625, 0.625, 0.4166, 0.3846, 0.5555, 1.25, 0.625, 0.625, 0.1562, 0.2083, 0.3846, 0.2631, 0.625, 0.8333, 0.3333, 1.25, 0.2380, 0.4545, 0.625, 0.5555, 0.4166, 0.8333, 0.2272, 0.8333, 0.2173, 0.2941, 0.625, 0.8333, 0.625, 1.25, 0.625, 0.8333, 0.2083, 0.1219, 0.1388, 0.2380, 0.4166, 0.3846, 0.625, 0.5, 0.5, 0.4166, 0.8333, 0.7142, 0.8333, 0.625, 0.5, 0.2083, 0.3333, 0.5, 1.25, 0.8333, 0.4166, 0.625, 0.1515, 0.625, 0.625, 0.7142, 0.8333, 0.8333, 0.25, 0.625, 0.3571, 0.25, 0.5, 0.7142, 0.4166, 0.8333, 0.4545, 0.4545, 0.5555, 0.3125, 0.625, 0.3125, 0.25, 0.8333, 0.625, 0.3333, 0.5, 0.625, 0.4166, 0.5555, 0.1785, 0.25, 0.4545, 0.7142, 0.2380, 0.5555, 0.2777, 0.8333, 0.5555, 0.8333, 1.0, 0.4545, 0.4166, 0.8333, 0.2777, 0.1562, 0.2777, 0.25, 0.3846, 0.8333, 0.2173, 0.5, 0.3846, 0.4166, 0.7142, 1.25, 0.4166, 1.0, 1.0, 0.2631, 0.625, 0.2083, 0.625, 0.1666, 1.25, 0.5, 0.5555, 0.4166, 0.3125, 0.7142, 0.625, 0.7142, 0.7142, 0.5555, 0.2083, 0.3333, 0.2631, 0.8333, 0.1470, 0.4166, 0.8333, 0.8333, 0.1666, 0.2941, 0.4166, 0.8333, 0.3571, 0.4166, 0.25, 0.1785, 0.2380, 0.1851, 0.2777, 0.4166, 0.25, 0.7142, 0.25, 0.7142, 0.2941, 0.1041, 1.0, 0.4166, 0.1351, 0.2173, 0.4166, 1.0, 0.0847, 0.1562, 0.7142, 0.625, 0.5, 0.1785, 0.5, 0.4166, 0.8333, 0.3333, 0.7142, 0.1923, 0.5, 0.5555, 0.2777, 0.7142, 0.2941, 0.3333, 0.5, 0.5, 0.3571, 0.2083, 0.3846, 0.625, 1.25, 0.625, 0.4166, 0.7142, 0.3333, 0.8333, 0.8333, 0.0847, 1.0, 0.3571, 0.8333, 0.8333, 0.5, 0.7142, 0.3846, 0.625, 0.1, 0.7142, 0.3333, 0.7142, 1.25, 0.1562, 0.25, 0.3846, 0.5555, 0.4166, 0.5555, 0.625, 0.2777, 0.625, 1.0, 0.3125, 0.4545, 0.4166, 0.4166, 0.5555, 0.4166, 0.1470, 0.625, 0.625, 0.8333, 0.5, 0.7142, 0.3333, 0.1851, 0.7142, 0.4545, 0.1785, 0.0450, 0.7142, 0.3846, 0.3333, 0.625, 1.0, 0.3846, 0.8333, 0.625, 0.7142, 1.0, 0.3333, 0.4545, 0.625, 0.2173, 0.3125, 0.1785, 0.7142, 0.625, 0.1785, 0.3571, 0.625, 0.625, 0.8333, 0.7142, 0.8333, 0.625, 0.1724, 0.2777, 0.0537, 0.625, 0.1923, 0.8333, 0.8333, 0.2777, 1.0, 0.8333, 0.7142, 0.4166, 0.8333, 0.5, 0.625, 0.3571, 0.7142, 0.4166, 0.8333, 0.8333, 0.1515, 0.4166, 0.3125, 0.7142, 0.5555, 0.3571, 0.2083, 0.1612, 0.7142, 0.5, 0.2272, 0.3846, 0.625, 0.8333, 0.4545, 0.5, 0.4166, 0.2941, 0.0925, 0.2380, 0.625, 1.0, 0.1470, 0.8333, 0.3571, 0.3125, 1.25, 0.7142, 0.8333, 0.7142, 0.3333, 0.5555, 0.2380, 0.8333, 0.2, 0.3571, 0.5555, 0.5555, 0.4545, 0.7142, 0.8333, 0.3846, 1.0, 0.7142, 0.5, 0.3125, 0.4545, 0.5, 0.8333, 1.25, 0.5, 0.8333, 1.25, 0.4545, 0.7142, 0.625, 0.0877, 0.2777, 0.2941, 0.1428, 1.0, 0.2380, 0.1, 1.25, 0.5, 0.625, 0.5, 0.3125, 1.25, 0.625, 0.2777, 0.625, 0.4166, 0.625, 0.4545, 0.625, 0.5555, 1.0, 0.4166, 1.0, 0.5, 0.2777, 0.5, 0.3333, 1.25, 0.8333, 0.625, 0.4166, 0.4166, 0.5555, 0.4166, 0.5555, 1.25, 0.4166, 0.3846, 0.5555, 0.2631, 0.5555, 0.5, 0.3846, 0.7142, 0.4545, 0.8333, 1.25, 0.4166, 0.25, 0.2, 0.0625, 0.3333, 0.4545, 0.8333, 0.4545, 0.5, 0.1923, 0.7142, 0.4166, 0.2777, 0.1785, 0.8333, 0.8333, 0.3125, 0.2, 0.3571, 1.0, 0.2941, 0.625, 0.2941, 0.7142, 0.1724, 0.4166, 0.625, 0.5555, 1.25, 0.625, 0.625, 0.625, 0.5555, 0.7142, 0.3333, 0.2777, 1.0, 0.5555, 0.8333, 0.5, 0.4166, 0.2083, 0.7142, 0.5555, 0.5, 1.25, 0.5555, 0.625, 0.8333, 0.3125, 0.7142, 0.4166, 0.0515, 0.625, 0.7142, 0.5, 0.625, 0.625, 1.25, 0.8333, 0.1388, 0.2777, 0.4166, 0.625, 0.7142, 0.5, 0.5, 0.7142, 1.25, 1.0, 1.0, 0.625, 1.25, 1.0, 0.5, 0.625, 0.8333, 0.125, 0.0961, 0.0877, 0.5555, 0.4166, 0.3846, 0.3846, 0.4166, 0.5, 0.4166, 0.1562, 0.8333, 0.2083, 0.7142, 0.3125, 0.625, 0.2173, 0.4166, 0.4166, 0.25, 0.625, 0.5555, 0.2941, 0.8333, 0.8333, 0.625, 0.8333, 0.4166, 0.2941, 0.1219, 1.0, 0.2941, 0.4166, 0.7142, 0.625, 0.5555, 0.4166, 0.4166, 0.2083, 0.3125, 0.3571, 1.25, 0.5, 0.8333, 0.2631, 0.7142, 0.5, 0.8333, 0.625, 0.3125, 0.625, 0.8333, 0.625, 1.0, 0.8333, 0.7142, 0.8333, 0.625, 0.7142, 0.1219, 0.1562, 0.5, 0.3125, 0.5, 0.4545, 0.4166, 0.625, 1.0, 0.625, 0.8333, 0.4166, 0.3125, 0.625, 0.4166, 0.4166, 0.3846, 0.1020, 0.5, 0.7142, 0.1851, 0.3571, 0.0909, 0.1851, 0.2272, 0.3125, 0.3125, 0.4166, 0.8333, 0.7142, 0.4166, 0.1724, 0.5555, 1.0, 0.4166, 0.8333, 0.2777, 0.625, 0.625, 0.625, 0.8333, 0.5, 0.625, 0.2173, 0.5555, 0.4545, 0.625, 0.8333, 0.7142, 0.4545, 0.625, 0.7142, 0.3333, 0.625, 0.2083, 0.5, 1.0, 0.0961, 0.2380, 1.0, 0.4545, 0.3846, 0.5555, 0.4545, 0.4166, 0.625, 0.7142, 0.4166, 1.0, 0.625, 1.25, 0.2, 0.8333, 0.2083, 0.2173, 0.4545, 0.0431, 0.5, 0.1923, 0.3846, 1.0, 0.7142, 1.0, 0.5, 0.7142, 0.0806, 0.3846, 0.5555, 0.0980, 0.8333, 0.5555, 0.625, 0.2631, 0.1351, 0.8333, 0.3125, 0.8333, 0.625, 0.4166, 0.2941, 1.0, 0.8333, 1.25, 0.5555, 0.8333, 1.0, 0.8333, 0.625, 0.2, 0.3333, 0.3571, 0.625, 0.8333, 0.3571, 0.625, 0.8333, 0.8333, 0.5555, 0.5555, 0.625, 0.1666, 0.8333, 0.3125, 0.3571, 0.4166, 0.8333, 0.625, 0.2083, 0.3571, 0.7142, 0.625, 0.3571, 0.625, 0.8333, 0.3333, 0.625, 0.3125, 0.2777, 1.25, 0.625, 0.7142, 0.1562, 0.8333, 1.0, 0.7142, 0.625, 0.5, 0.4166, 0.2777, 0.625, 0.625, 0.2272, 0.7142, 0.1666, 0.3846, 0.625, 0.3125, 0.7142, 0.1136, 0.8333, 0.7142, 0.3846, 0.5555, 0.625, 0.625, 0.4545, 1.0, 0.5, 0.1388, 0.5, 0.625, 0.625, 1.0, 0.1785, 0.7142, 0.4545, 0.8333, 0.5555, 0.8333, 0.1388, 0.7142, 0.625, 0.4166, 0.3333, 0.5, 1.25, 0.2272, 0.625, 0.1470, 0.1666, 0.1315, 0.5555, 0.625, 0.5, 0.7142, 0.5, 0.4166, 1.25, 1.0, 0.5555, 0.625, 0.625, 1.0, 0.3571, 0.4545, 0.625, 0.3333, 0.625, 0.3125, 0.2941, 0.7142, 0.5, 0.4545, 0.4545, 0.625, 0.2777, 0.625, 0.1785, 0.25, 0.4166, 0.5555, 0.625, 0.5, 0.3125, 0.625, 0.625, 0.625, 0.8333, 0.3125, 0.4166, 0.7142, 0.4166, 0.3125, 1.0, 0.3571, 0.5555, 0.3846, 0.1562, 0.3571, 0.3846, 0.5555, 0.625, 0.2941, 0.4545, 1.0, 0.625, 0.3333, 0.5, 0.1785, 0.1851, 1.25, 0.5555, 0.625, 0.625, 0.5, 0.8333, 0.2777, 0.5, 1.25, 0.625, 0.5555, 0.2083, 0.25, 0.7142, 0.5, 0.8333, 0.4166, 0.3571, 0.4166, 0.3333, 0.25, 0.4545, 0.2777, 0.0943, 0.625, 0.5555, 0.625, 0.5555, 0.5555, 0.1612, 0.8333, 0.4545, 0.5555, 0.625, 0.3125, 0.7142, 1.25, 0.5555, 0.4545, 0.5, 0.8333, 0.25, 0.8333, 0.7142, 0.4166, 0.4166, 0.5, 0.7142, 0.2777, 0.5555, 0.3846, 0.2, 0.2941, 0.3333, 0.3333, 0.8333, 0.3125, 0.25, 1.0, 0.625, 0.3333, 0.2083, 1.25, 0.7142, 0.1785, 0.7142, 0.7142, 0.2083, 0.7142, 0.3125, 0.2941, 0.1785, 0.3846, 0.5, 0.625, 0.3125, 0.7142, 0.1020, 0.2380, 0.7142, 0.7142, 0.7142, 0.4166, 0.4545, 0.3571, 0.625, 0.5555, 0.8333, 0.3333, 0.2777, 0.5, 0.3125, 0.625, 0.8333, 0.8333, 0.25, 0.5, 0.3571, 0.3846, 0.5, 0.7142, 0.5555, 0.4545, 0.2173, 0.3333, 0.625, 0.625, 0.625, 0.8333, 0.3333, 0.2941, 0.3125, 0.1282, 0.4166, 0.8333, 0.8333, 0.625, 0.4545, 0.8333, 0.4166, 0.625, 0.625, 0.5, 0.625, 0.3571, 0.25, 0.3125, 0.3846, 0.625, 1.0, 0.4166, 0.7142, 1.25, 0.625, 0.7142, 0.5555, 0.3846, 0.4166, 0.625, 0.1923, 0.5, 0.7142, 0.4166, 0.25, 0.8333, 0.5555, 0.3571, 0.1428, 1.25, 0.1315, 0.8333, 1.25, 0.8333, 0.4166, 0.3846, 0.7142, 1.25, 0.7142, 1.0, 0.625, 1.0, 0.625, 1.0, 1.25, 0.4166, 1.0, 0.8333, 0.625, 0.5555, 1.0, 1.0, 0.8333, 0.2631, 0.3125, 0.3125, 0.4166, 0.625, 0.4545, 0.1219, 0.3571, 0.3125, 1.0, 0.5555, 0.7142, 0.8333, 0.1562, 0.4166, 0.625, 1.25, 1.25, 0.3846, 0.8333, 0.0833, 0.1315, 1.25, 1.0, 0.5, 0.8333, 0.3333, 0.8333, 0.4166, 0.3125, 0.7142, 0.625, 0.4545, 0.625, 0.2, 1.25, 0.2941, 1.25, 0.3333, 0.8333, 0.625, 0.3571, 0.5555, 0.1388, 0.625, 0.2380, 0.625, 0.3571, 0.4166, 0.5, 0.2083, 1.0, 0.2941, 0.7142, 1.25, 0.3846, 0.5555, 0.3125, 0.2083, 0.625, 0.5, 0.1666, 1.25, 1.0, 0.3846, 0.8333, 0.1923, 0.3125, 0.7142, 0.7142, 0.8333, 0.2380, 0.4545, 0.2777, 0.3846, 0.8333, 0.3846, 0.625, 0.5555, 0.4166, 0.3333, 1.25, 0.4545, 0.8333, 0.7142, 0.625, 0.625, 0.7142, 0.625, 0.5, 0.7142, 0.1923, 1.0, 0.5, 0.5555, 0.7142, 0.7142, 0.5555, 0.1388, 0.8333, 0.1351, 0.8333, 0.625, 0.1162, 0.5555, 0.5555, 1.0, 0.2777, 0.1219, 0.5555, 0.5555, 0.3846, 1.25, 0.8333, 0.4166, 0.4166, 0.4166, 0.2941, 1.0, 0.7142, 0.3333, 0.3125, 0.3846, 1.0, 0.4166, 0.7142, 0.4545, 1.25, 0.7142, 0.7142, 0.625, 0.5, 0.4166, 0.5555, 0.3125, 0.5555, 0.3846, 1.25, 0.3125, 0.2083, 0.7142, 0.3846, 0.5, 0.8333, 0.7142, 0.4545, 0.625, 1.0, 0.5, 0.7142, 0.4166, 1.0, 0.4545, 0.1612, 0.5555, 0.4545, 0.7142, 0.8333, 0.3846, 0.4166, 0.1923, 0.1388, 1.0, 0.1, 0.1470, 0.5, 0.4166, 1.0, 0.3846, 0.1562, 0.8333, 0.3571, 0.8333, 0.625, 0.1666, 0.2631, 0.5555, 0.8333, 0.4166, 0.25, 0.1923, 0.8333, 0.625, 0.5555, 0.7142, 0.625, 0.7142, 0.8333, 0.8333, 1.25, 0.625, 0.4545, 1.0, 0.3125, 1.0, 0.7142, 0.4166, 0.1282, 0.625, 0.2083, 0.5555, 0.625, 0.8333, 0.8333, 0.7142, 0.2173, 0.2272, 0.4166, 0.3333, 0.3125, 0.3125, 0.625, 0.2777, 0.2777, 0.4166, 0.2941, 0.8333, 0.1923, 0.3846, 0.1428, 0.4545, 0.3571, 0.625, 1.25, 0.5, 0.3846, 0.7142, 0.1470, 0.3125, 0.5555, 0.3846, 0.625, 0.8333, 0.4545, 0.25, 0.625, 0.625, 0.8333, 0.5555, 0.2631, 0.1666, 0.8333, 0.5, 0.625, 0.4166, 0.4545, 0.625, 0.625, 0.8333, 0.7142, 1.0, 0.25, 0.625, 1.25, 0.4545, 0.5, 0.5555, 0.625, 0.625, 0.625, 0.8333, 0.5, 0.1111, 0.4166, 0.625, 0.8333, 0.2173, 0.4166, 1.0, 0.2380, 0.7142, 0.7142, 0.4166, 0.8333, 0.3846, 1.25, 0.4166, 0.625, 0.3333, 0.1851, 0.4545, 0.625, 0.5, 0.625, 0.625, 1.0, 0.8333, 0.1428, 0.4545, 0.4166, 0.625, 0.2631, 0.625, 0.4166, 0.8333, 0.5555, 0.8333, 0.0375, 0.625, 0.4166, 0.625, 0.8333, 0.5555, 0.3571, 0.625, 1.0, 0.1, 0.2, 0.5555, 0.625, 0.5555, 0.7142, 0.625, 0.2173, 0.625, 0.5555, 0.7142, 0.625, 0.625, 0.7142, 0.7142, 0.8333, 1.0, 0.8333, 0.1851, 0.3125, 0.1612, 0.1388, 0.5555, 0.1785, 0.8333, 0.7142, 0.7142, 0.625, 0.1923, 0.7142, 0.2083, 0.0806, 0.3125, 0.625, 0.1515, 0.3333, 0.2173, 0.3571, 0.2777, 0.5, 0.3333, 0.0609, 0.625, 0.7142, 0.5555, 0.3571, 0.1388, 0.5, 0.8333, 0.4166, 0.5555, 0.625, 1.25, 0.0549, 0.8333, 0.7142, 0.2380, 0.7142, 0.4166, 0.3571, 0.625, 0.8333, 0.8333, 0.3846, 0.3846, 0.625, 0.3571, 0.2083, 0.0757, 0.625, 0.2380, 1.0, 0.7142, 0.625, 0.625, 0.3125, 0.5555, 0.7142, 0.2777, 0.625, 0.4545, 0.4545, 0.8333, 0.2173, 0.5, 0.7142, 0.5, 0.8333, 0.3125, 1.25, 0.625, 0.1666, 0.625, 1.25, 0.25, 0.625, 0.3571, 1.0, 0.5, 0.625, 0.1562, 0.3125, 0.3846, 1.25, 0.8333, 0.4545, 1.0, 0.5, 0.4166, 0.625, 0.2380, 1.25, 1.25, 0.4545, 0.625, 0.625, 0.5555, 0.8333, 0.1388, 0.3571, 0.625, 0.2272, 0.4166, 1.0, 0.5555, 0.8333, 0.25, 0.2631, 0.5555, 0.3846, 0.7142, 0.25, 0.8333, 0.3846, 1.25, 0.5, 0.4545, 0.7142, 0.3333, 0.625, 0.25, 1.0, 0.25, 0.8333, 0.2941, 0.2777, 0.625, 0.2777, 0.3333, 0.8333, 0.7142, 0.2, 0.5555, 0.3846, 0.7142, 0.3333, 0.5555, 0.7142, 0.3571, 0.3125, 0.625, 0.3333, 0.5, 0.2083, 0.0462, 0.3846, 0.3846, 0.5555, 0.2083, 0.3846, 0.8333, 0.5555, 0.5, 0.3846, 0.8333, 0.25, 0.8333, 0.625, 0.625, 0.3125, 0.4166, 0.3571, 0.5555, 1.0, 1.0, 1.0, 0.7142, 0.625, 0.4166, 0.7142, 0.5555, 1.25, 0.625, 0.25, 1.25, 1.25, 0.5555, 0.7142, 0.625, 0.5555, 0.625, 0.1785, 0.3125, 0.625, 0.625, 0.7142, 0.3333, 0.4545, 0.625, 0.1041, 0.3125, 0.4545, 0.5, 0.3846, 0.2941, 0.4166, 0.5, 0.2631, 1.25, 0.5, 0.2631, 0.3571, 0.5555, 0.2631, 0.7142, 0.3125, 0.3571, 0.3125, 0.3846, 0.625, 0.1219, 1.0, 0.5, 0.7142, 1.25, 0.2380, 0.625, 0.5555, 0.4166, 0.5, 1.25, 0.4166, 0.5, 0.4166, 0.8333, 0.2380, 1.0, 0.5555, 0.0892, 0.625, 0.3333, 0.25, 0.3571, 1.25, 1.0, 0.5555, 0.5555, 1.25, 0.4166, 0.2083, 0.5, 0.8333, 0.1666, 0.4166, 0.2941, 1.0, 0.625, 0.7142, 0.7142, 0.625, 0.8333, 0.0781, 0.5, 0.1724, 0.4166, 0.1388, 0.4166, 0.625, 0.625, 0.1851, 0.3125, 0.5555, 0.4166, 0.8333, 0.4166, 1.0, 0.4166, 0.8333, 0.7142, 0.3333, 0.4166, 0.4166, 0.4166, 0.3333, 0.625, 0.7142, 0.625, 0.2777, 0.2631, 0.4545, 0.2083, 0.625, 0.2941, 1.25, 0.8333, 0.5555, 0.3125, 0.1388, 1.25, 0.2631, 0.4545, 0.8333, 0.2631, 0.4545, 1.0, 0.3125, 0.5, 0.8333, 0.2272, 0.8333, 0.5555, 0.0833, 0.1785, 0.7142, 0.2272, 0.3333, 0.3333, 0.3571, 1.0, 0.8333, 0.25, 0.25, 0.5555, 0.2631, 0.625, 0.625, 1.0, 0.25, 0.625, 0.25, 0.7142, 0.2173, 0.625, 0.8333, 0.8333, 0.5555, 0.4166, 0.3125, 0.4166, 0.625, 0.5, 0.4166, 0.8333, 0.3846, 0.7142, 1.0, 0.25, 0.625, 0.8333, 0.8333, 0.2173, 0.8333, 0.2272, 1.0, 0.625, 0.2380, 0.625, 0.625, 0.2941, 0.7142, 0.7142, 0.3846, 0.3125, 0.1666, 0.8333, 0.25, 1.25, 0.3846, 0.7142, 0.4545, 0.4545, 0.5555, 0.7142, 0.5555, 0.0694, 0.625, 0.7142, 0.4545, 0.2272, 0.625, 0.4545, 0.4545, 0.8333, 0.1470, 1.25, 0.2777, 0.4166, 0.5, 0.2083, 1.25, 0.2941, 0.5555, 0.4545, 0.3125, 0.8333, 0.3125, 0.7142, 0.625, 0.25, 0.5, 0.625, 0.2777, 0.5555, 0.625, 0.625, 0.5555, 0.5, 0.5555, 0.2631, 0.3125, 0.25, 0.25, 0.2083, 0.625, 0.625, 0.625, 0.8333, 0.3125, 0.5555, 0.5, 0.3571, 0.25, 0.8333, 0.4166, 0.5, 0.5555, 0.625, 0.2941, 0.7142, 0.4166, 1.0, 0.7142, 0.4166, 0.4166, 0.4166, 0.3125, 0.4545, 0.2272, 0.1470, 0.2941, 0.4166, 0.625, 1.25, 0.3333, 0.4166, 0.2083, 0.3333, 0.5555, 0.4166, 0.625, 0.7142, 0.625, 0.1428, 0.3571, 0.625, 1.0, 0.2941, 0.8333, 0.4166, 0.8333, 0.8333, 0.3125, 0.3125, 0.125, 0.7142, 0.8333, 0.4166, 0.8333, 0.5555, 0.3333, 0.8333, 0.4545, 0.5555, 0.5555, 0.25, 0.4545, 0.5555, 0.4545, 0.5555, 0.8333, 0.625, 0.1923, 0.5555, 0.3125, 0.4166, 0.3571, 1.0, 0.5, 0.625, 0.8333, 0.5, 0.7142, 0.4166, 0.7142, 0.0632, 0.7142, 0.1041, 0.3125, 0.3571, 0.7142, 0.4166, 0.1190, 0.1041, 0.3125, 0.25, 0.625, 0.2272, 0.625, 0.4545, 0.4166, 0.1612, 0.2272, 0.5555, 0.2, 0.5, 0.625, 0.4166, 0.3125, 0.3333, 0.5555, 0.2777, 0.625, 0.8333, 0.8333, 0.3333, 0.25, 0.8333, 0.8333, 0.3125, 0.3846, 0.5, 0.1666, 0.1136, 0.8333, 0.625, 0.5, 0.8333, 0.8333, 0.7142, 0.4166, 0.8333, 0.7142, 0.3125, 1.0, 0.3125, 0.1562, 0.4545, 0.4166, 0.5555, 0.8333, 0.4545, 0.3571, 0.4166, 0.625, 1.0, 0.8333, 0.3125, 0.1724, 0.4545, 0.625, 0.3125, 0.3571, 1.0, 1.25, 0.3571, 0.2272, 0.625, 0.7142, 1.0, 0.625, 1.25, 0.8333, 0.625, 0.7142, 0.625, 0.4166, 0.625, 0.1470, 0.625, 0.4545, 0.4166, 0.7142, 0.8333, 0.5, 0.1724, 0.7142, 0.2777, 0.1136, 0.625, 0.5555, 0.3125, 0.7142, 0.3846, 0.5555, 0.1428, 0.8333, 0.2380, 0.7142, 0.5, 0.25, 0.8333, 0.625, 0.4166, 0.3125, 0.7142, 0.625, 1.0, 0.2380, 0.0925, 0.625, 0.3846, 0.1785, 0.5555, 1.0, 0.7142, 0.8333, 0.5, 0.2941, 1.25, 0.25, 0.7142, 0.3333, 0.5555, 0.625, 0.8333, 0.2777, 1.0, 0.625, 0.0806, 1.25, 0.2941, 0.625, 0.3125, 0.5, 0.7142, 0.8333, 0.625, 1.0, 0.5, 0.4545, 0.7142, 0.1515, 0.7142, 0.625, 0.7142, 0.5555, 0.4166, 0.3571, 0.3846, 0.7142, 0.625, 0.5, 0.625, 0.2631, 0.3333, 0.3571, 0.3846, 1.0, 0.25, 0.625, 0.625, 0.4166, 0.1428, 0.5555, 0.0625, 0.8333, 0.1923, 0.7142, 0.8333, 0.7142, 1.0, 0.1724, 0.625, 0.7142, 0.625, 0.2941, 0.625, 0.3125, 0.1562, 0.5, 0.5555, 0.4166, 0.8333, 0.7142, 0.7142, 0.4166, 0.625, 0.4166, 0.4166, 0.4166, 0.7142, 0.5, 0.5555, 0.0961, 0.1562, 1.0, 0.4166, 0.3333, 0.4166, 0.1388, 0.5555, 0.4545, 0.7142, 0.625, 0.4166, 0.3125, 0.7142, 0.625, 0.8333, 0.4166, 0.1111, 1.0, 0.4166, 0.1923, 0.5555, 0.25, 0.2, 0.5, 0.1923, 0.2631, 0.1219, 0.5, 0.2941, 0.2777, 0.2380, 0.7142, 0.1724, 0.8333, 0.625, 0.8333, 0.8333, 0.7142, 0.3846, 0.2777, 0.8333, 0.4166, 0.0892, 1.0, 1.0, 1.0, 0.1724, 0.625, 0.3571, 0.625, 0.3125, 1.0, 0.8333, 0.2777, 0.1612, 0.1724, 0.3125, 0.4166, 0.5, 0.3846, 0.625, 0.25, 0.7142, 0.3333, 0.7142, 0.25, 0.5555, 0.4166, 0.2941, 0.3571, 1.0, 1.0, 0.1086, 0.7142, 0.8333, 0.625, 0.5, 0.3125, 0.625, 0.625, 0.625, 0.2083, 0.1724, 0.3846, 0.625, 0.1785, 0.3571, 0.4166, 0.4166, 1.0, 0.5555, 0.7142, 0.4545, 1.0, 0.2, 0.3571, 0.25, 1.0, 0.1470, 0.1851, 0.4166, 0.7142, 0.4166, 0.2, 0.7142, 0.3571, 0.3846, 0.8333, 0.2083, 0.2, 0.2272, 0.2941, 1.0, 0.5, 0.5, 0.1351, 0.625, 0.625, 0.625, 0.3125, 0.7142, 0.1162, 0.0833, 0.4166, 0.3125, 0.4166, 0.3125, 0.5555, 1.0, 0.625, 0.2941, 0.0561, 0.8333, 0.4166, 0.5555, 0.3125, 0.625, 0.625, 0.625, 0.2631, 0.4545, 0.1428, 0.7142, 0.2083, 1.0, 1.0, 1.0, 0.4166, 0.4166, 0.625, 0.3125, 0.7142, 0.4166, 0.7142, 0.625, 0.5, 0.0847, 0.5, 0.625, 0.7142, 0.8333, 1.25, 0.1785, 0.8333, 0.4166, 0.3571, 0.4545, 0.5555, 0.1923, 0.8333, 0.3333, 0.3125, 0.8333, 0.2941, 0.3125, 0.3125, 0.3125, 0.7142, 0.625, 0.1515, 0.0943, 0.7142, 0.4545, 0.1785, 0.3333, 0.2941, 0.4545, 0.3125, 0.7142, 0.2272, 0.1315, 0.5555, 0.7142, 0.1724, 0.5, 0.625, 0.0961, 0.7142, 0.8333, 0.4545, 0.4166, 0.8333, 0.625, 0.1136, 0.1470, 0.5555, 0.625, 0.1851, 0.625, 0.5555, 0.5555, 0.625, 0.5555, 0.8333, 0.1724, 0.5555, 0.625, 0.3846, 0.2941, 0.5555, 0.4166, 0.8333, 0.3846, 0.1785, 0.5555, 0.625, 0.5555, 0.625, 0.5555, 0.1428, 0.5, 0.2631, 0.5, 0.625, 0.3846, 0.3571, 0.625, 0.5, 0.8333, 0.8333, 0.1562, 0.3333, 0.8333, 0.4545, 0.7142, 0.2941, 0.625, 0.2777, 0.2083, 1.0, 0.2941, 1.0, 0.625, 0.3571, 0.3125, 1.0, 0.1851, 0.4545, 0.625, 0.8333, 0.25, 0.2083, 0.625, 0.625, 0.2631, 0.8333, 0.625, 0.625, 0.4166, 0.4166, 0.2380, 0.2380, 1.0, 0.5555, 0.1785, 0.5, 0.3125, 0.625, 0.1351, 0.625, 0.1136, 0.4166, 0.7142, 0.8333, 0.8333, 0.2083, 0.7142, 1.25, 0.4545, 0.7142, 0.8333, 0.3846, 0.625, 0.5, 0.2083, 0.3125, 0.1562, 1.25, 0.8333, 0.8333, 0.0625, 1.0, 0.5555, 0.5555, 0.8333, 0.8333, 0.2777, 0.5555, 0.8333, 0.625, 0.5, 0.7142, 0.625, 0.3125, 0.625, 0.7142, 1.0, 0.625, 0.5555, 0.5555, 0.5555, 0.4545, 0.3333, 0.625, 0.4166, 0.4166, 1.25, 1.0, 0.3571, 1.0, 0.3571, 0.2777, 0.625, 0.3125, 0.4166, 0.5, 0.4166, 0.3125, 0.25, 0.8333, 0.4545, 0.1063, 0.625, 0.8333, 0.1190, 0.4166, 0.25, 0.625, 0.4166, 0.25, 0.625, 0.3125, 0.3846, 0.4166, 0.8333, 0.2083, 0.3846, 1.25, 0.8333, 0.625, 0.625, 0.625, 0.1562, 0.625, 0.5555, 0.3125, 0.2941, 0.2777, 0.625, 0.3571, 0.8333, 0.625, 0.25, 0.5555, 0.5555, 0.1562, 0.5, 0.3333, 0.1086, 0.3333, 1.0, 0.625, 0.3333, 0.7142, 1.0, 0.625, 1.25, 0.8333, 0.625, 1.25, 0.5555, 0.3125, 0.3846, 1.0, 0.7142, 0.3846, 0.625, 0.8333, 0.8333, 0.3125, 1.0, 0.3846, 0.1785, 0.8333, 0.4545, 0.1666, 0.8333, 0.8333, 0.3125, 1.25, 0.1562, 0.3333, 0.4166, 1.0, 0.8333, 0.5555, 0.5, 0.4545, 0.5, 0.3125, 0.1470, 0.2083, 0.625, 0.625, 0.625, 0.8333, 0.7142, 0.3846, 0.4166, 0.3125, 0.625, 0.4545, 0.4545, 0.7142, 0.2083, 1.0, 0.5, 0.2173, 0.8333, 0.1282, 0.3846, 0.5, 0.4545, 1.0, 0.25, 0.4545, 0.625, 0.7142, 0.2777, 1.25, 0.1515, 0.625, 0.7142, 0.3846, 0.4166, 0.3571, 0.1219, 0.3333, 0.2173, 0.3846, 0.1851, 0.8333, 0.2631, 0.8333, 0.625, 0.4166, 0.3333, 0.2631, 0.5, 0.1388, 1.0, 0.5, 0.8333, 0.2777, 0.625, 0.2631, 0.7142, 0.4166, 0.625, 0.5555, 0.625, 0.3333, 1.0, 0.3125, 0.3571, 0.4166, 1.0, 0.5, 0.4166, 0.4166, 0.2380, 1.0, 0.25, 0.125, 0.4166, 0.5555, 0.625, 0.5555, 0.25, 0.125, 0.2941, 0.625, 0.2941, 0.25, 0.5, 0.7142, 0.3333, 0.7142, 0.5, 0.2941, 0.8333, 0.625, 0.5555, 0.625, 0.1562, 0.4545, 0.8333, 0.8333, 0.8333, 0.3846, 0.4545, 0.3333, 0.1282, 0.5, 0.1388, 0.4545, 1.25, 0.3125, 0.7142, 0.625, 0.25, 0.625, 0.3571, 0.3846, 1.0, 0.7142, 1.25, 0.7142, 0.625, 0.4166, 0.2272, 0.5555, 0.5555, 0.7142, 0.3125, 0.0781, 0.3333, 0.2777, 0.0649, 0.7142, 0.8333, 0.625, 0.625, 0.7142, 0.2777, 0.5555, 0.2380, 0.1388, 0.8333, 0.5555, 0.3571, 0.8333, 1.0, 0.4166, 0.3125, 0.7142, 0.8333, 0.4545, 0.8333, 0.8333, 0.8333, 0.625, 0.1923, 0.4166, 0.1666, 0.4166, 0.8333, 0.7142, 0.7142, 1.0, 0.625, 0.7142, 0.8333, 0.625, 0.4166, 0.7142, 0.625, 0.625, 0.8333, 0.25, 0.2941, 0.0781, 1.0, 0.3125, 0.1923, 0.4166, 0.2777, 0.5, 0.625, 0.5, 0.1562, 0.3125, 0.8333, 0.7142, 0.1351, 0.7142, 0.1388, 0.8333, 0.8333, 0.4166, 0.8333, 0.2941, 1.25, 0.3125, 0.5555, 0.7142, 0.4166, 1.0, 0.1351, 1.0, 0.3846, 0.8333, 0.8333, 0.8333, 0.3571, 0.7142, 1.25, 0.8333, 0.625, 0.1785, 1.0, 0.1851, 0.4545, 0.8333, 0.0806, 0.7142, 0.3125, 1.0, 0.4166, 0.3846, 0.7142, 0.4166, 0.7142, 0.1851, 0.5555, 1.0, 0.3846, 0.625, 0.3846, 0.2777, 0.5, 0.2173, 0.7142, 0.1388, 0.5, 0.7142, 0.8333, 1.25, 0.625, 0.5, 0.4545, 0.625, 0.625, 0.8333, 0.4166, 1.0, 0.8333, 0.625, 0.2631, 0.25, 0.625, 0.4166, 0.3125, 0.3125, 0.2631, 0.1428, 0.625, 0.8333, 0.1351, 0.4166, 0.7142, 0.3333, 0.1063, 0.2631, 0.625, 0.3571, 0.1515, 0.4166, 0.625, 0.8333, 0.3571, 0.2631, 0.4545, 0.8333, 0.625, 0.625, 0.625, 1.25, 0.8333, 0.625, 0.7142, 0.8333, 0.3846, 0.8333, 0.4166, 0.5, 0.4166, 0.1111, 0.2777, 0.5555, 0.2272, 0.625, 0.625, 0.7142, 0.3333, 0.2, 0.1612, 0.3846, 0.0520, 1.0, 0.1388, 0.5, 0.4166, 1.25, 0.1851, 0.4545, 0.8333, 0.2380, 0.8333, 0.2, 0.4166, 0.3571, 1.0, 0.625, 0.7142, 0.7142, 0.3125, 0.625, 0.3846, 0.2272, 0.3125, 0.7142, 0.7142, 0.2777, 0.625, 0.8333, 0.8333, 0.1785, 0.4166, 0.1086, 0.4166, 0.7142, 0.625, 0.8333, 0.25, 0.1219, 0.8333, 0.4166, 0.8333, 0.625, 0.8333, 0.5555, 0.4545, 0.3125, 0.625, 0.625, 0.625, 0.4166, 0.1923, 0.3125, 0.5555, 0.625, 0.3125, 0.0595, 0.5, 0.5555, 0.5555, 0.7142, 0.3571, 0.625, 0.2631, 0.625, 0.4545, 0.625, 0.3571, 0.1515, 0.5555, 0.1785, 0.1785, 0.3125, 0.7142, 0.5, 0.8333, 0.1470, 0.2941, 0.4545, 0.7142, 0.3333, 1.0, 0.5555, 0.2941, 0.3571, 0.625, 0.4166, 0.5555, 0.3333, 0.625, 0.2083, 0.625, 0.25, 0.7142, 0.2272, 1.25, 0.625, 0.3333, 0.3846, 0.1470, 0.7142, 0.4166, 0.625, 0.8333, 0.5555, 0.625, 0.8333, 0.2380, 0.1851, 0.5555, 0.625, 0.3125, 0.4166, 0.0746, 0.4166, 0.2631, 0.1515, 0.1612, 0.4166, 0.2, 0.7142, 0.3125, 0.0833, 0.5555, 0.2083, 0.625, 0.1388, 0.5, 0.1612, 0.625, 0.2083, 0.0980, 0.2272, 0.4166, 0.7142, 0.0961, 0.4545, 0.625, 1.0, 0.3125, 0.7142, 0.625, 0.7142, 0.3125, 0.7142, 0.1612, 0.3846, 0.3571, 0.625, 0.25, 1.25, 0.3125, 0.2777, 0.8333, 0.4545, 0.25, 0.5555, 0.3125, 0.3846, 0.2173, 0.2777, 0.625, 0.4166, 0.4545, 0.8333, 0.8333, 0.4166, 0.8333, 1.25, 0.25, 0.7142, 0.7142, 0.2083, 0.8333, 0.3125, 0.4166]).cuda()
    #Class_CrossEntropyLoss = nn.CrossEntropyLoss(weight=classes_weight).cuda()
    Class_CrossEntropyLoss = nn.CrossEntropyLoss().cuda()
    #TODO set weights for unbalanced label.
    #channel_weight = torch.Tensor([1.0, 0.05]).cuda()

    Official_tripletloss = nn.TripletMarginWithDistanceLoss(margin=margin, reduction='sum', distance_function=lambda x,y: 1.0 - F.cosine_similarity(x,y)).cuda()
    #Channel_CrossEntropyLoss = nn.CrossEntropyLoss(weight=channel_weight).cuda()
    Channel_CrossEntropyLoss = nn.CrossEntropyLoss().cuda()
    Triple_loss = TripleLoss().cuda()
    
    optimizer_D1G = optim.SGD([ {'params': D1.parameters()}, {'params': G.parameters()}], lr=lr_init, momentum=0.8)
    #scheduler_D1G = None
    lambda1 = lambda epoch: 0.95 ** epoch
    scheduler_D1G = torch.optim.lr_scheduler.LambdaLR(optimizer_D1G, lr_lambda=[lambda1, lambda1])
    #scheduler_D1G = MultiStepLR(optimizer_D1G, milestones=[100,200,300,400,500,600,700], gamma=0.5)
    optimizer_D2 = optim.SGD([ {'params': D2.parameters()}, {'params': G.parameters()} ], lr=lr_init, momentum=0.8)
    
    
    train_dataset = MyDataset(mode='train', batch_size=batch_size)
    train_dataloader = DataLoader(train_dataset,batch_size = 1, shuffle=False, drop_last=True, num_workers=30 )
    
    G.train()
    D1.train()
    D2.train()
    for epoch in range(epochs):
        train_one_epoch(train_dataloader, G, D1, D2, Class_CrossEntropyLoss, Triple_loss,  optimizer_D1G, optimizer_D2, epoch, batch_size, Channel_CrossEntropyLoss,Official_tripletloss,scheduler_D1G,  args)


def train_one_epoch(train_dataloader, G, D1, D2, Class_CrossEntropyLoss,  Triple_loss,  optimizer_D1G, optimizer_D2, epoch, batch_size, Channel_CrossEntropyLoss, Official_tripletloss, scheduler_D1G, args):
    cur_idx = 0
    for batch_id,(x,y) in enumerate(train_dataloader):
        print ("batch_id:")
        print (batch_id)
        idx = epoch * len(train_dataloader) + batch_id
        
        x = x.squeeze()
        x = x.transpose(dim0=0, dim1=1)
        x = x.cuda()
        y = y.squeeze()
        y = y.cuda()
        generator_feature = G(x)
        generator_feature = generator_feature.transpose(dim0=0, dim1=1)
        generator_feature = generator_feature.unsqueeze(dim=1)
        print ('generator_feature.shape:')
        print (generator_feature.shape)
        print ('generator_feature:')
        #print (generator_feature)
        spk_embedding, class_logit = D1(generator_feature)
        spk_embedding = spk_embedding.squeeze()
        print ('spk_embedding:')
        #print (spk_embedding)
        print ('input_y:')
        #print (y.shape)
        print (y)
        pred_y = torch.argmax(class_logit,1)
        print ('pred_y:')
        print (pred_y)
        print ('class_logit:')
        print (class_logit)
        accuracy = y.eq(pred_y).sum().item() / len(y)
        print ('train accuracy:')
        print (accuracy)
        writer.add_scalar("accuracy/train", accuracy,idx)
        
        #train G,D1
        optimizer_D1G.zero_grad()
        #compute loss
        #TODO: use less to more classes for classification
        Ls = Class_CrossEntropyLoss(class_logit,y)
        Ls.backward()
        #spk_embedding = normalize(spk_embedding)
        #embedding_ap, embedding_an = Triple_loss(spk_embedding, y)
        #Lt = Official_tripletloss(spk_embedding, embedding_ap, embedding_an)
        ##Lt.backward()
        #L_d1 = Ls + Lt
        #L_d1.backward()

        #writer.add_scalar('loss/triple', Lt.item(),idx)
        writer.add_scalar('loss/softmax', Ls.item(),idx)
        #writer.add_scalar('loss/triple+softmax', L_d1.item(),idx)

        #TODO  maybe not same with Eq.11. Detail: D2'G network parameters shoud be update in condition of same input with D1.

        for name,param in G.named_parameters():
            writer.add_scalar(name + "/weight_mean"  , torch.mean(param.data).item(),idx)
            writer.add_scalar(name + "/grad_mean"  , torch.mean(param.grad).item(),idx)

        for name,param in D1.named_parameters():
            writer.add_scalar(name + "/weight_mean"  , torch.mean(param.data).item(),idx)
            writer.add_scalar(name + "/grad_mean"  , torch.mean(param.grad).item(),idx)


        #update gradients.  TODO:MIN?
        optimizer_D1G.step()


        ##train D2
        #generator_feature = G(x)
        #generator_feature = generator_feature.transpose(dim0=0, dim1=1)
        #generator_feature = generator_feature.unsqueeze(dim=1)
        #pred_channel = D2(generator_feature)
        ###get channel_label
        ##TODO get channel label gracefully.
        #channel_label = []
        #for i in y:
        #    if i.item() <= 379:
        #        channel_label.append(0)    
        #    else:
        #        channel_label.append(1)    
        #channel_label = torch.from_numpy(np.array(channel_label)).cuda()
        #print ('--------------------------')
        #print ('pred_channel.shape:')
        #print (pred_channel.shape)
        #print (pred_channel)
        #print ('channel_label.shape:')
        #print (channel_label.shape)
        #print (channel_label)
        ##TODO Eq.10 is max loss. check.
        #L_d2 = 0 - Channel_CrossEntropyLoss(pred_channel,channel_label)
        ##L_d2 = Channel_CrossEntropyLoss(pred_channel,channel_label)
        #optimizer_D1G.zero_grad()
        #optimizer_D2.zero_grad()
        #L_d2.backward()
        #writer.add_scalar('loss/channel', L_d2.item(),idx)
        ##update gradients. TODO:max?
        #optimizer_D2.step()
        

        # example input & output
        #input = torch.randn(8, 500, 64, 1)
        #input = input.squeeze(dim=-1)
        #input = input.transpose(dim0=0, dim1=1)
        #hx = input.new_zeros(input.size(1), projection_size, requires_grad=False)
        #cx = input.new_zeros(input.size(1), hidden_size, requires_grad=False)
        #state = [hx, cx]
        #G = Generator(input_size=input_size, hidden_size=hidden_size, projection_size=projection_size)
        #output = G(input, state)
        #output_LSTM = output.transpose(dim0=0, dim1=1)
        #output = output_LSTM.unsqueeze(dim=1)
        #D = Discriminator_speaker()
        #output = D(output)
        #D2 = Discriminator_channel()
        #output = D2(output_LSTM)

        print ('lr:')
        writer.add_scalar("lr"  , scheduler_D1G.get_last_lr()[0],idx)
    scheduler_D1G.step()
       
if __name__ == '__main__':
    args = parse_args()
    main(args)
    writer.close()
    #dataset = MyDataset()
    #dataset.__getitem__(0)

