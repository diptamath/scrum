from __future__ import absolute_import, division, print_function
import cv2
import numbers
import math
import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import pytorch_ssim
import torchvision

from tqdm import tqdm

import MegaDepth 


from tensorboardX import SummaryWriter
# writer = SummaryWriter('runs/main/ph3_v4')

device = torch.device("cuda:1")


# depthnet=MegaDepth.__dict__['HourGlass']("MegaDepth/checkpoints/test_local/best_generalization_net_G.pth")
depthnet=MegaDepth.__dict__['HourGlass']()
depthnet[4] = nn.Conv2d(in_channels=64,out_channels=4,kernel_size=3,stride=1,padding=1)

depthnet.to(device)

# vgg16 = torchvision.models.vgg16()
# vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22])
# vgg16_conv_4_3.to(device)
# for param in vgg16_conv_4_3.parameters():
#     param.requires_grad = False

feed_width = 512*2
feed_height = 384*2

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma=None, dim=2):
        super(GaussianSmoothing, self).__init__()
        # default sigma value
        if sigma is None :
            sigma = 0.3*((kernel_size-1)/2 -1) + 0.8

        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        
        self.conv1 = GaussianSmoothing(channels=3, kernel_size = 75) 
        self.conv2 = GaussianSmoothing(channels=3, kernel_size = 45) 
        self.conv3 = GaussianSmoothing(channels=3, kernel_size = 25) 
        
        self.pad1 = nn.ReflectionPad2d(37)
        self.pad2 = nn.ReflectionPad2d(22)
        self.pad3 = nn.ReflectionPad2d(12)

       

    def forward(self,x,disp):
        blur1 = self.conv1(self.pad1(x))
        blur2 = self.conv2(self.pad2(x))
        blur3 = self.conv3(self.pad3(x))

        mask1 = torch.stack([disp[:,0,:,:],disp[:,0,:,:],disp[:,0,:,:]],dim=1)
        mask2 = torch.stack([disp[:,1,:,:],disp[:,1,:,:],disp[:,1,:,:]],dim=1)
        mask3 = torch.stack([disp[:,2,:,:],disp[:,2,:,:],disp[:,2,:,:]],dim=1)
        mask4 = torch.stack([disp[:,3,:,:],disp[:,3,:,:],disp[:,3,:,:]],dim=1)

        comp1 = mask1*blur1
        comp2 = mask2*blur2
        comp3 = mask3*blur3 
        comp4 = mask4*x

        y = comp1 + comp2 + comp3 + comp4 

        return y, blur1, blur2, blur3 ,comp1 , comp2 , comp3 , comp4

class bokehDataset(Dataset):
    
    def __init__(self, csv_file,root_dir, transform=None):
        
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        bok = pil.open(self.root_dir + self.data.iloc[idx, 0][1:]).convert('RGB')
        org = pil.open(self.root_dir + self.data.iloc[idx, 1][1:]).convert('RGB')
            
        bok = bok.resize((feed_width, feed_height), pil.LANCZOS)
        org = org.resize((feed_width, feed_height), pil.LANCZOS)

        

        if self.transform : 
            bok = self.transform(bok)
            org = self.transform(org)

        return (bok,org)

def smooth_loss(im):
    return torch.mean(torch.abs(im[:,:,1:,:]-im[:,:,:-1,:])) + torch.mean(torch.abs(im[:,:,:,1:]-im[:,:,:,:-1]))     

transform1 = transforms.Compose(
    [
    transforms.ToTensor(),
])


transform2 = transforms.Compose(
    [
    transforms.RandomHorizontalFlip(p=1),
    transforms.ToTensor(),
])


transform3 = transforms.Compose(
    [
    transforms.RandomVerticalFlip(p=1),
    transforms.ToTensor(),
])


trainset1 = bokehDataset(csv_file = './train.csv', root_dir = '/media/data2/saikat/bokeh_data',transform = transform1)
trainset2 = bokehDataset(csv_file = './train.csv', root_dir = '/media/data2/saikat/bokeh_data',transform = transform2)
trainset3 = bokehDataset(csv_file = './train.csv', root_dir = '/media/data2/saikat/bokeh_data',transform = transform3)

# batch size changed
trainloader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([trainset1,trainset2,trainset3]), batch_size=1,
                                          shuffle=True, num_workers=2)

testset = bokehDataset(csv_file = './test.csv',  root_dir = '/media/data2/saikat/bokeh_data', transform = transform1)
testloader = torch.utils.data.DataLoader(testset, batch_size=2,
                                          shuffle=False, num_workers=2)


gen = generator()

print ('generator loaded!!')        
gen = gen.to(device)
gen.eval()

from gridnet import GridNet
refinenet = GridNet(27,3).to(device)

# learning_rate = 0.001
learning_rate = 0.001
optimizer = optim.Adam( list(refinenet.parameters()), lr=learning_rate, betas=(0.9, 0.999))
                                                            

sm = nn.Softmax(dim=1)
ssim_loss = pytorch_ssim.SSIM()
MSE_LossFn = nn.MSELoss()

def mae_loss(frame1, frame2):
    return torch.norm(frame1 - frame2,p=1)/frame1.numel()

def train(dataloader):
    running_l1_loss = 0
    running_ms_loss = 0
    running_loss = 0
    for i,data in enumerate(tqdm(dataloader),0) : 
        bok , org = data 
        bok , org = bok.to(device) , org.to(device)
        with torch.no_grad():
            out = sm(depthnet(org))

        # print(out.shape)

        bok_pred, b1, b2, b3 , c1, c2, c3, c4 = gen(org,out)
        res_bok = refinenet(torch.cat((bok_pred,b1,b2,b3,org,c1,c2,c3,c4),dim=1))
        bok_final = bok_pred + res_bok

        l1_loss = mae_loss(bok_final,bok)
        ms_loss = 1-ssim_loss(bok_final, bok)

        running_l1_loss += l1_loss.item()
        running_ms_loss += ms_loss.item()
        #running_smooth_loss += sm_loss.item()

        loss = l1_loss + 0.1*ms_loss
        print (loss.item())
        running_loss += loss.item()	

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print ('Batch: ',i,'/',len(dataloader),' Loss:', loss.item())

        if (i%2000==0):
            # torch.save(depthnet.state_dict(), './models/main/mega_models3_v6/depth-'+str(epoch)+'-'+str(i)+'.pth')
            torch.save(refinenet.state_dict(), './models/main/mega_models3_v6/ref-'+str(epoch)+'-'+str(i)+'.pth')


    print (running_l1_loss/len(dataloader))    
    print (running_ms_loss/len(dataloader))  
    print (running_loss/len(dataloader))  


def val(dataloader):
    running_l1_loss = 0
    running_ms_loss = 0

    with torch.no_grad():
        for i,data in enumerate(tqdm(dataloader),0) : 
            bok , org = data 
            bok , org = bok.to(device) , org.to(device)
            out = sm(depthnet(org))
            bok_pred, b1, b2, b3 , c1, c2, c3, c4 = gen(org,out)
            res_bok = refinenet(torch.cat((bok_pred,b1,b2,b3,org,c1,c2,c3,c4),dim=1))
            bok_final = bok_pred + res_bok
       

            l1_loss = mae_loss(bok_final,bok)
            ms_loss = 1-ssim_loss(bok_final, bok)
            
            running_l1_loss += l1_loss.item()
            running_ms_loss += ms_loss.item()


    print ('Validation l1 Loss: ',running_l1_loss/len(dataloader))   
    print ('Validation ssim Loss: ',running_ms_loss/len(dataloader))


os.makedirs('./models/main/mega_models3_v6/', exist_ok=True)

# depthnet.load_state_dict(torch.load('./models/main/mega_models2/depth-1.pth',map_location=device))
depthnet.load_state_dict(torch.load('../BokehGen/phase3_best.pth',map_location=device))
depthnet.eval()


start_ep = 0
for epoch in range(start_ep,40):    
    print (epoch)
    
    with torch.no_grad():
        val(testloader)

    train(trainloader) 


#################################################################################



