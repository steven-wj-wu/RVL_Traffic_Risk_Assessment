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






classes = read_classes('classes.txt')
num_frame = 20
imgSize = (224,224)


if __name__ == '__main__':


	flag=0


	read_baseUrl = "/media/leo/C0EC4D32EC4D244E/TRA_split_20/dataset_20"
	save_baseUrl = "/media/leo/C0EC4D32EC4D244E/TRA_split_20/My_test_dataset"

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
					frame = cv2.imread(frame_path)


					if f_num>-1:
						frame = cv2.resize(frame, (224,224))
						#flow_img = do_opticalflow(prev_frame,frame
						series.append(frame)

					#print(len(series))

					if len(series)==num_frame:
						os.makedirs(os.path.join(save_baseUrl,label,label+"_"+"test"+"_"+str(sp_clip_count).zfill(6))) #ex: .../high/hign_0000001
						for i in range (num_frame):
							save_path = os.path.join(save_baseUrl,label,label+"_"+"test"+"_"+str(sp_clip_count).zfill(6)+"/"+"image_"+str(i+1).zfill(5)+".jpg")
							cv2.imwrite(save_path,series[i])
						print('save clip:',label,"/",str(sp_clip_count).zfill(6))
						print(save_path)
						sp_clip_count = sp_clip_count+1



