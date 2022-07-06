from torch.utils.data import Dataset,DataLoader
from torch import nn

  
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size per GPU')
    parser.add_argument('--epochs', default=100, type=int, help='epochs')

class MyDataset(Dataset)：
    def __init__(self):
        print ("init dataset")

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

