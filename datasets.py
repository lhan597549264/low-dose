import glob
import random
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        
        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    
class testDataset(Dataset):
    def __init__(self,root,transforms_=None):
        super().__init__() 
        self.transform = transforms.Compose(transforms_)
        self.root=root
        self.dir=os.path.join(self.root,"test","A")
        self.testlist=os.listdir(self.dir)
    def __getitem__(self, index):
        test_dir=os.path.join(self.dir,self.testlist[index])
        
        item=self.transform(Image.open(test_dir))
        return {"name":self.testlist[index],"pic":item}
    def __len__(self):
        return len(self.testlist)
transforms_ = [ transforms.Resize(512),  
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)) ]

