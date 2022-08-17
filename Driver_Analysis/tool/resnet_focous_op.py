import argparse
import os
import time
import pickle as pickle
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from model_op import Model
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

    best = torch.load('./ckpts/op.tar')  #my model
    model.load_state_dict(best['state_dict'])
 
    root_path='/home/leo/Desktop/3dresnet_video/series_image/dataset'
    labels = listdir(root_path)

    for lable in labels:
        lable_path =  os.path.join(root_path,lable)
        video_names = listdir(lable_path)

        for name in video_names:
            video_path =  os.path.join(lable_path,name)
            clips = sorted(listdir(video_path))

            first_clip_path = os.path.join(video_path,clips[0])
            file_name = clips[0].split('.')
            first_frame = cv2.imread(first_clip_path)
            first_heat_img = do_weight(first_frame,first_frame,model)
            save_name = '/home/leo/Desktop/3dresnet_video/series_image/dataset/'+lable+'/'+name+'/'+file_name[0]+'_weight'+'.jpg'
            print(save_name)
            cv2.imwrite(save_name,first_heat_img)

            for i in range(9):
                pre_image_name="image_"+str(i+1).zfill(5)+".jpg"
                cur_image_name="image_"+str(i+2).zfill(5)+".jpg"
                save_name = "image_"+str(i+2).zfill(5)
                file_name = clips[i+1].split('.')
                prev_frame = cv2.imread(os.path.join(video_path,pre_image_name))
                next_frame = cv2.imread(os.path.join(video_path,cur_image_name))
                heat_img = do_weight(prev_frame,next_frame,model)
                save_name = '/home/leo/Desktop/3dresnet_video/series_image/dataset/'+lable+'/'+name+'/'+save_name+'_weight'+'.jpg'
                print(save_name)
                cv2.imwrite(save_name,heat_img)


def do_weight(prev_frame,next_frame,model):

        ori_frame = cv2.resize(next_frame, (224,224))
        prev_frame =  cv2.resize(prev_frame, (320, 192), interpolation=cv2.INTER_CUBIC)
        prev_gray = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)

        next_frame =  cv2.resize(next_frame, (320, 192), interpolation=cv2.INTER_CUBIC)
        next_gray = cv2.cvtColor(next_frame,cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_gray,next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        next_frame = next_frame.astype('float32')/255.0
        model_input = np.concatenate((next_frame,flow),axis=2)
        model_input = model_input.transpose(2, 0, 1)
        model_input = model_input[None, ...]  
        model_input = np.ascontiguousarray(model_input)
        model_input = torch.from_numpy(model_input)

        test_loader = DataLoader(
                    model_input,
                    batch_size=32, shuffle=False,
                    num_workers=0,
                    pin_memory=True
                    )

        outputs = predict(test_loader, model)

        mask = outputs[0] #normalize
        if mask.max()!=0:
            mask = mask/mask.max()
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

        mask_frame = cv2.resize(mask_img, (224,224))
        #blurred = cv2.GaussianBlur(ori_frame,(25,25),0)
        blurred = cv2.multiply(ori_frame,(0.5,0.5,0.5,0.5))
        blurred[mask_frame>10]=ori_frame[mask_frame>10]

        

        

        return blurred

def do_blur(prev_frame,next_frame,model):

        ori_frame = cv2.resize(next_frame, (224,224))
        prev_frame =  cv2.resize(prev_frame, (320, 192), interpolation=cv2.INTER_CUBIC)
        prev_gray = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)

        next_frame =  cv2.resize(next_frame, (320, 192), interpolation=cv2.INTER_CUBIC)
        next_gray = cv2.cvtColor(next_frame,cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_gray,next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        next_frame = next_frame.astype('float32')/255.0
        model_input = np.concatenate((next_frame,flow),axis=2)
        model_input = model_input.transpose(2, 0, 1)
        model_input = model_input[None, ...]  
        model_input = np.ascontiguousarray(model_input)
        model_input = torch.from_numpy(model_input)

        test_loader = DataLoader(
                    model_input,
                    batch_size=32, shuffle=False,
                    num_workers=0,
                    pin_memory=True
                    )

        outputs = predict(test_loader, model)

        mask = outputs[0] #normalize
        if mask.max()!=0:
            mask = mask/mask.max()
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

        mask_frame = cv2.resize(mask_img, (224,224))
        blurred = cv2.GaussianBlur(ori_frame,(25,25),0)
        blurred[mask_frame>20]=ori_frame[mask_frame>20]

        

        

        return blurred

            

def do_optical(prev_frame,next_frame):

        prev_gray = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_frame,cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_gray,next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        hsv = np.zeros_like(next_frame)
        hsv[...,1] = 255
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        return bgr

def do_saliency(prev_frame,next_frame,model):

        ori_frame = next_frame
        prev_frame =  cv2.resize(prev_frame, (320, 192), interpolation=cv2.INTER_CUBIC)
        prev_gray = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)

        next_frame =  cv2.resize(next_frame, (320, 192), interpolation=cv2.INTER_CUBIC)
        next_gray = cv2.cvtColor(next_frame,cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_gray,next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        next_frame = next_frame.astype('float32')/255.0
        model_input = np.concatenate((next_frame,flow),axis=2)
        model_input = model_input.transpose(2, 0, 1)
        model_input = model_input[None, ...]  
        model_input = np.ascontiguousarray(model_input)
        model_input = torch.from_numpy(model_input)

        test_loader = DataLoader(
                    model_input,
                    batch_size=32, shuffle=False,
                    num_workers=0,
                    pin_memory=True
                    )

        outputs = predict(test_loader, model)

        mask = outputs[0] #normalize
        if mask.max()!=0:
            mask = mask/mask.max()
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

        mask_frame = cv2.resize(mask_img, (224,224))
        mask_frame = cv2.applyColorMap(mask_frame,cv2.COLORMAP_JET)

        heat_img = get_heatmap(ori_frame,mask_frame)

        return heat_img





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

def get_heatmap(ori_img,mask_img):
    overlay=ori_img.copy()
    alpha=0.2
    cv2.rectangle(overlay, (0, 0), (ori_img.shape[1], ori_img.shape[0]), (255, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, ori_img, 1-alpha, 0, ori_img)
    cv2.addWeighted(mask_img, alpha, ori_img, 1-alpha, 0, ori_img) 
    return ori_img


if __name__ == '__main__':
    main()
