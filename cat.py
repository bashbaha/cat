from torch.utils.data import Dataset,DataLoader
from torch import nn

  
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size per GPU')
    parser.add_argument('--epochs', default=100, type=int, help='epochs')

class MyDataset(Dataset)：
    def __init__(self, mode = 'train'):
        print ("init dataset")
        self.mode = mode
        self.datafile = '../data/train_demo.list'
        self.file_ids = open(self.datafile,'r').readlines()
        self.width = 500
        self.height = 64
 	self.sample_rate = 8000
    def __getitem__(self, idx):
        f,label = self.file_ids[idx].strip().split(" ")
        wav,sr = torchaudio.load(f,normalize=False)
        assert sr == self.sample_rate
        wav = wav / 1.0
        feature = fbank(wav, dither=1,high_freq=-200, low_freq=64, htk_compat=True,  num_mel_bins=self.height, sample_frequency=self.sample_rate, use_energy=False, window_type='hamming')
        feature_len = len(feature)
        if self.mode == "train": #random start pieces
            rand_start = random.randint(0,feature_len - self.width)
            print ('debug: rand_start: ' + str(rand_start))
            feature = feature[rand_start : rand_start + self.width]
        else: #fixed feature for test
            feature = feature[0 : self.width]

        #normalize
        std,mu = torch.std_mean(feature,dim=0)
        feature = (feature - mu) / (std + 1e-5)

        feature = torch.unsqueeze(feature, dim=0)
        label = torch.LongTensor([int(label)])

        return feature,label 

    def __len__(self):
        return len(self.file_ids)

class Generator():
    def __init__(self):
        print ("init Generator")
        
class Discriminator_speaker():
    def __init__(self):
        print ("init Discriminator_speaker")

class Discriminator_channel():
    def __init__(self):
        print ("init Discriminator_channel")




def main(args):
    num_classes = 251
    batch_size = args.batch_size
    lr_init = args.lr
    epochs = args.epochs
    
    D1 = Discriminator_speaker()
    D2 = Discriminator_channel()
    G = Generator()
    
    Class_CrossEntropyLoss = nn.CrossEntropyLoss()
    AdversarialLoss = nn.BCELoss()
    
    for epoch in range(epochs):
        train_one_epoch(train_dataloader, G, D1, D2,  Class_CrossEntropyLoss, AdversarialLoss, optimizer_G, optimizer_D1, optimizer_D2, epoch, batch_size, args)

def train_one_epoch(train_dataloader, G, D1, D2, Class_CrossEntropyLoss, AdversarialLoss, optimizer_G, optimizer_D1, optimizer_D2, epoch, batch_size, args):
    for batch_id,(x,y) in enumerate(train_dataloader):
        #compute loss
        #update gradients
        #save log & model
        pass
    
       
if __name__ == '__main__':
    args = parse_args()
    main(args)

