import os
import sys
import cv2 
import numpy as np
from tqdm import tqdm
from pathlib import Path  
import random


import torch
import glob
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision
import pathlib
import PIL
from PIL import Image, ImageOps
import pickle as pickle
import flow_vis


from saliency.data_load import ImageList
from saliency.model_op import Model

def read_classes(file_path):

	fp = open(file_path, "r")
	classes = fp.readline()
	classes = classes. split(",")
	fp.close()
	last = classes[3].split("\n")
	classes[3]=last[0]
	print(classes)

	return classes


def predict_saliency(valid_loader, model):

	# switch to evaluate mode
	model.eval()
	outputs = []
	targets = [] 

	for i, input in enumerate(valid_loader):
		input = input.cuda()
		input_var = torch.autograd.Variable(input)
		# compute output
		output = model(input_var)
		outputs.append(output.data.cpu().numpy().squeeze(1))

	outputs = np.concatenate(outputs)
	return outputs

""""""
def do_saliency(prev_frame,next_frame,model):

		
		ori_frame = next_frame
		prev_frame =  cv2.resize(prev_frame, (320, 192), interpolation=cv2.INTER_CUBIC)
		prev_gray = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)

		next_frame =  cv2.resize(next_frame, (320, 192), interpolation=cv2.INTER_CUBIC)
		next_gray = cv2.cvtColor(next_frame,cv2.COLOR_BGR2GRAY)
		flow = cv2.calcOpticalFlowFarneback(prev_gray,next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0) #calc op

		####to hsv image
		#hsv = np.zeros_like(prev_frame)
		#hsv[...,1] = 255
		
		#mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
		#hsv[...,0] = ang*180/np.pi/2
		#hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
		#bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
			

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

		outputs = predict_saliency(test_loader, model)

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

		return mask_frame


def do_opticalflow(prev_frame,next_frame):

	ori_frame = next_frame
	#prev_frame =  cv2.resize(prev_frame, (224, 224), interpolation=cv2.INTER_CUBIC)
	prev_gray = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)

	#next_frame =  cv2.resize(next_frame, (224, 224), interpolation=cv2.INTER_CUBIC)
	next_gray = cv2.cvtColor(next_frame,cv2.COLOR_BGR2GRAY)
	hsv = np.zeros_like(prev_frame)
	hsv[...,1] = 255
	#flow = cv2.optflow.calcOpticalFlowSparseToDense(prev_gray,next_gray,sigma = 0.07)
	flow = cv2.calcOpticalFlowFarneback(prev_gray,next_gray, None, 0.5, 3, 10, 3, 5, 1.1, 0) #calc op
	mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
	hsv[...,0] = ang*180/np.pi/2
	hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
	bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
	#bgr = cv2.resize(bgr,imgSize)
	flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=False)
	

	
	return flow_color




classes = read_classes('classes.txt')
num_frame = 19
imgSize = (224,224)


if __name__ == '__main__':


	flag=0


	read_baseUrl = "/media/leo/C0EC4D32EC4D244E/TRA_split_20_ori/rgb/validation"
	save_baseUrl = "/media/leo/C0EC4D32EC4D244E/TRA_split_20_ori/flow/validation"

	TRA_Labels = os.listdir(read_baseUrl)
	print(TRA_Labels)

	for idx,label in enumerate(TRA_Labels):
		sp_clip_count=0
		label_url=os.path.join(read_baseUrl,label)
		print(label_url)
		
		clip_list = sorted(os.listdir(label_url))
		#read each frame in clips and save in split folder 
		for num,clip in enumerate(clip_list):
			frame_url = os.path.join(label_url,clip)
			frame_list = os.listdir(frame_url) 
			series=[]
			for f_num,frame in enumerate(frame_list):
				if not (frame == 'n_frames'):
					frame_path = os.path.join(frame_url,frame)
					#print(frame_path)
					frame = cv2.imread(frame_path)

					if f_num>0:

						#flow_img = do_opticalflow(prev_frame,frame)
						flow = do_opticalflow(prev_frame,frame)
						#flow=cv2.resize(flow, (224, 224), interpolation=cv2.INTER_CUBIC)
						prev_frame = frame.copy()

						
						series.append(flow)

					prev_frame = frame.copy()
					#print(len(series))

					if len(series)==num_frame:
						os.makedirs(os.path.join(save_baseUrl,label,label+"_"+"valid"+"_"+str(sp_clip_count).zfill(6))) #ex: .../high/hign_0000001
						for i in range (num_frame):
							save_path = os.path.join(save_baseUrl,label,label+"_"+"valid"+"_"+str(sp_clip_count).zfill(6)+"/"+"image_"+str(i+1).zfill(5)+".jpg")
							cv2.imwrite(save_path,series[i])
						print('save clip:',label,"/",str(sp_clip_count).zfill(6))
						print(save_path)
						sp_clip_count = sp_clip_count+1



