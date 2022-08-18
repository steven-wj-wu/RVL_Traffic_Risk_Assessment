import os
import sys
import cv2 
import numpy as np
from pathlib import Path  

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
print(classes)

num_frame = 9
f=1
imgSize = (224,224)
list_num=0

if __name__ == '__main__':

	read_baseUrl = "/media/leo/C0EC4D32EC4D244E/TRA_split_20/TRA_split_train_TSM"
	save_baseUrl = "/media/leo/C0EC4D32EC4D244E/tra_single"

	#make split dir
	if(not os.path.exists(save_baseUrl)):
		print('make label folder')
		for i in range(len(classes)):
			os.makedirs(save_baseUrl+"/"+str(classes[i])) # ex: /media/leo/C0EC4D32EC4D244E/TRA_split/high
	
	
	#read daset label
	TRA_Labels = os.listdir(read_baseUrl)
	print(TRA_Labels)
	#read video clips in each label
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
				if frame=='n_frames':
					print('skip')
				else:
					frame_path = os.path.join(frame_url,frame)
					frame = cv2.imread(frame_path)
					series.append(frame)
					
					if len(series)==num_frame:
						os.makedirs(os.path.join(save_baseUrl,label,label+"_"+"test"+"_"+str(sp_clip_count).zfill(6))) #ex: .../high/hign_0000001
						for i in range (num_frame):
							save_path = os.path.join(save_baseUrl,label,label+"_"+"test"+"_"+str(sp_clip_count).zfill(6)+"/"+"image_"+str(i+1).zfill(5)+".jpg")
							cv2.imwrite(save_path,series[i])
						series=[]
						print('save clip:',label,"/",str(sp_clip_count).zfill(6))
						sp_clip_count = sp_clip_count+1



				"""
				if f_num == -1: #skip first rgb
					print(1)
				else:
					frame_path = os.path.join(frame_url,frame)
					print(frame_path)
					if(f_num+1)%num_frame==0:
						#print(f_num)
						frame = cv2.imread(frame_path)
						#frame = cv2.resize(frame,imgSize)
						series.append(frame)
				"""

			"""
			os.makedirs(os.path.join(save_baseUrl,label,label+"_"+"train"+"_"+str(sp_clip_count).zfill(6))) #ex: .../high/hign_0000001
			for i in range (19):
				save_path = os.path.join(save_baseUrl,label,label+"_"+"train"+"_"+str(sp_clip_count).zfill(6)+"/"+"image_"+str(i+1).zfill(5)+".jpg")
				cv2.imwrite(save_path,series[i])
			print('save clip:',label,"/",str(sp_clip_count).zfill(6))
			sp_clip_count = sp_clip_count+1
			"""
		
				
	
	






