import argparse
import os
import time
import pickle as pickle
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from model import Model
from data_load import ImageList
import random
import warnings
import logging
import numpy as np
import json
import cv2
from os import listdir


warnings.simplefilter("ignore")


def main():
    #global args, best_score
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.manual_seed(2017)
    torch.cuda.manual_seed(2017)
    random.seed(2017)
    np.random.seed(2017)

    model = Model()
    model = model.cuda()
    params = model.parameters()

    cudnn.benchmark = True
 
   
    video_list = sorted(listdir('./BDDA_test'))
    video_path = './BDDA_test/'

  
    best = torch.load('./ckpts/best.tar')  #my model
    model.load_state_dict(best['state_dict'])

    
    for i in range(len(video_list)):
        video_name = video_path+video_list[i]
        video_num = video_list[i].split('.')
        cap = cv2.VideoCapture(video_name)
        print('video:',video_name)

        c = 0
        num=1
        f = 14
        flag=0
        while (cap.isOpened()):
            ret, video_frame = cap.read()
            if ret == True:
                if c%f==0:
                    

                    video_frame =  cv2.resize(video_frame, (320, 192), interpolation=cv2.INTER_CUBIC)                 
                    model_input = video_frame.astype('float32')/255.0
                
                    
                    model_input = model_input.transpose(2, 0, 1)
                    model_input = model_input[None, ...]  
                    model_input = np.ascontiguousarray(model_input)
                    model_input = torch.from_numpy(model_input)
         
             
                    test_loader = DataLoader(
                    model_input,
                    batch_size=32, shuffle=False,
                    num_workers=0,
                    pin_memory=True)
            
              
                    outputs = predict(test_loader, model)
                    
                
                    mask = outputs[0] #normalize
                    #if mask.max()!=0:
                        #mask = mask/mask.max()
                    mask = mask*255
                
            
                    mask_img = np.zeros((192, 320,1), np.uint8)
                    col=0
                    row=0
                    for i in mask:
                        for j in i:
                            mask_img[col,row,0]=j
                            row=row+1
                        row=0	
                        col=col+1


                    mask_img = cv2.resize(mask_img, (64,36))
                    cv2.imwrite('./cdnn_write/'+video_num[0]+'_'+str(num)+'.jpg',mask_img)

                    num=num+1
                c=c+1
                
            
            else:
                break

def predict(valid_loader, model):

    # switch to evaluate mode
    model.eval()
    outputs = []
    targets = [] 

    

    for i, input in enumerate(valid_loader):

        
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        # compute output
        output = model(input_var)
        outputs.append(output.data.cpu().numpy().squeeze(1))

    outputs = np.concatenate(outputs)
    return outputs


if __name__ == '__main__':
    main()
