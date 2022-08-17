from torch.utils.data import Dataset
import imageio as io
import cv2
import torch
from scipy.ndimage import filters
from numpy import *
import scipy.io as sio
import os
import numpy as np


np.set_printoptions(threshold=np.inf)
# shape = (180, 320)

def transform(x, y):
    if np.random.uniform() < 0.5:
        x = x[:, ::-1]
        y = y[:, ::-1]
    return x, y


def getLabel(mask_name):
    #fixdatafile = ('./fixdata/fixdata' + str(vid_index) + '.mat')
    #data = sio.loadmat(fixdatafile)

    #fix_x = data['fixdata'][frame_index - 1][0][:, 3]
    #fix_y = data['fixdata'][frame_index - 1][0][:, 2]
    """
    mask = np.zeros((720, 1280), dtype='float32')

    for i in range(len(fix_x)):
        mask[fix_x[i], fix_y[i]] = 1

    mask = filters.gaussian_filter(mask, 40)
    mask = np.array(mask, dtype='float32')
    # mask = cv2.resize(mask, (320, 192), interpolation=cv2.INTER_CUBIC)
    mask = mask.astype('float32') / 255.0

    if mask.max() == 0:
        print (mask.max())
        # print img_name
    else:
        mask = mask / mask.max()
    """
    #print(mask_name)

    mask = cv2.imread(mask_name,0)
    mask = np.array(mask, dtype='float32')
    mask = cv2.resize(mask, (320, 192), interpolation=cv2.INTER_AREA)
    mask = mask / 255.0

    if mask.max() == 0:
        print(mask.max())
    else:
        mask=mask/mask.max()

    #print(np.max(mask))

    return mask

class ImageList(Dataset):
    def __init__(self, root, imgs, for_train=False,for_valid=False,for_test=False):
        self.root = root
        self.imgs = imgs
        self.for_train = for_train
        self.for_valid = for_valid 
        self.for_test = for_test

    def __getitem__(self, index):
        
        img_name = self.imgs[index]
        #print(len(self.imgs))
        if len(self.imgs) > index+1:
            next_img_name= self.imgs[index+1]
        else:
            next_img_name= self.imgs[index]  
        
        prev = img_name.split('_')
        next = next_img_name.split('_')
        if  prev[0] != next[0] :
            next_img_name = img_name
       
        image_name = os.path.join(self.root, img_name)
        next_image_name = os.path.join(self.root, next_img_name)
        #print(image_name)
        #print(next_image_name)
    
        img = io.imread(image_name)
        next_image = io.imread(next_image_name)
        img = cv2.resize(img, (320, 192), interpolation=cv2.INTER_CUBIC)
        next_image = cv2.resize(next_image, (320, 192), interpolation=cv2.INTER_CUBIC)
   
        
        prev_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_image,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray,next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        train_img = next_image.astype('float32')/255.0
        
	
        if self.for_train:
        #mask = getLabel(vid_index, frame_index)
           
            mask = getLabel('./media/train_op/mask_15/'+next_img_name) #train

        if self.for_valid:
    
            mask = getLabel('./media/valid/mask_15/'+next_img_name) #valid
        
        if self.for_test:
            mask = getLabel(self.root+'/mask/'+img_name) #test
            mask = getLabel(self.root+img_name)

        if self.for_train:
            train_img, mask= transform(train_img, mask)

        train =  train_img
        #use flow data
        #train =  np.concatenate((train_img,flow),axis=2)

        train = train.transpose(2, 0, 1)
        mask = mask[None, ...]
        



        train = np.ascontiguousarray(train)
        mask = np.ascontiguousarray(mask)
        
	
        
        #print (torch.from_numpy(img))
        #print (torch.from_numpy(mask))
     
        return torch.from_numpy(train), torch.from_numpy(mask)
        """
        img_name = self.imgs[index]
        #vid_index = int(img_name[0:2])
        #frame_index = int(img_name[3:9])

        image_name = os.path.join(self.root, img_name)
        img = io.imread(image_name)
        img = cv2.resize(img, (320, 192), interpolation=cv2.INTER_CUBIC)
        img = img.astype('float32')/255.0

        #mask = getLabel(vid_index, frame_index)
        mask = getLabel('./media/train/mask/'+img_name) #train
        #mask = getLabel(self.root+'/mask/'+img_name) #test
        #mask = getLabel(self.root+img_name)

        if self.for_train:
            img, mask = transform(img, mask)

        img = img.transpose(2, 0, 1)
        mask = mask[None, ...]
        

        img = np.ascontiguousarray(img)
        mask = np.ascontiguousarray(mask)
	
        
        #print (torch.from_numpy(img))
        #print (torch.from_numpy(mask))
     
        return torch.from_numpy(img), torch.from_numpy(mask)
        """

    def __len__(self):
        return len(self.imgs)



