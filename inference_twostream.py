import os
import sys
import time
import warnings
import numpy as np
import random
import logging
import json
import cv2
import PIL
from PIL import Image, ImageOps
import pickle as pickle
import matplotlib.pyplot as plt
import io
import time

import torch
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import PyQt5.QtGui as QtGui
from PyQt5.QtCore import *
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


from opts import parse_opts
from mean import get_mean, get_std
#from resnet_3d_old.model_c_attention import generate_model
from model_d import generate_model
from spatial_transforms_winbus import (
	Compose, Normalize, RandomHorizontalFlip, ToTensor, RandomVerticalFlip, 
	ColorAugment)
from temporal_transforms import LoopPadding, TemporalRandomCrop, TemporalCenterCrop

from saliency.data_load import ImageList
from saliency.model_op import Model
import flow_vis


class Qt(QWidget):
    def mv_Chooser(self):    
        opt = QFileDialog.Options()
        opt |= QFileDialog.DontUseNativeDialog
        fileUrl = QFileDialog.getOpenFileName(self,"Input Video", "../../long_drive_dataset","AVI (*.mp4)", options=opt)
        file_name_split = fileUrl[0].split('/')
        file_name = file_name_split[-1]
        file_name = file_name.replace('mp4','txt')
        print(file_name)

        return fileUrl[0]





def read_classes(file_path):

	fp = open(file_path, "r")
	classes = fp.readline()
	classes = classes. split(",")
	fp.close()
	last = classes[3].split("\n")
	classes[3]=last[0]

	return classes

classes = read_classes('classes.txt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def letterbox(img, resize_size, mode='square'):
	#QT information show set
	shape = [img.size[1],img.size[0]]  # current shape [height, width]
	new_shape = resize_size
	if isinstance(new_shape, int):
		ratio = float(new_shape) / max(shape)
	else:
		ratio = max(new_shape) / max(shape)  # ratio  = new / old
	ratiow, ratioh = ratio, ratio
	new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
	if mode == 'auto':  # minimum rectangle
		dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding
		dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding
	elif mode == 'square':  # square
		dw = (new_shape - new_unpad[0]) / 2  # width padding
		dh = (new_shape - new_unpad[1]) / 2  # height padding
	elif mode == 'rect':  # square
		dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
		dh = (new_shape[0] - new_unpad[1]) / 2  # height padding
	elif mode == 'scaleFill':
		dw, dh = 0.0, 0.0
		new_unpad = (new_shape, new_shape)
		ratiow, ratioh = new_shape / shape[1], new_shape / shape[0]

	top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
	left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
	img = img.resize(new_unpad,PIL.Image.ANTIALIAS)
	img = ImageOps.expand(img, border=(left,top,right,bottom), fill=(128,128,128))
	return img


class letter_img(transforms.Resize):
	#QT information show set
	def __init__(self, size, interpolation=Image.BILINEAR):
		assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
		self.size = size
		self.interpolation = interpolation
	def __call__(self, img):
		return letterbox(img, self.size)
	def __repr__(self):
		interpolate_str = _pil_interpolation_to_str[self.interpolation]
		return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)



def get_test_data(images, spatial_transform):
	#image transform
	images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
	# pre_src = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
	clip = [Image.fromarray(img) for img in images]
	clip = [img.convert('RGB') for img in clip]
	#spatial_transform.randomize_parameters()
	clip = [spatial_transform(img) for img in clip]
	clip = torch.stack(clip, 0)
	clip = torch.stack((clip,), 0)

	return clip



def do_saliency(prev_frame,next_frame,model):

		#get driver attention
		prev_gray = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)
		next_gray = cv2.cvtColor(next_frame,cv2.COLOR_BGR2GRAY)
		flow_start = time.perf_counter()
		flow = cv2.calcOpticalFlowFarneback(prev_gray,next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0) #calc op
		flow_end = time.perf_counter()
		flow_time = flow_end-flow_start
		flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=False)
		flow_color = cv2.resize(flow_color, (224,224))


		prev_frame =  cv2.resize(prev_frame, (320, 192), interpolation=cv2.INTER_CUBIC)
		prev_gray = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)
		next_frame =  cv2.resize(next_frame, (320, 192), interpolation=cv2.INTER_CUBIC)
		next_gray = cv2.cvtColor(next_frame,cv2.COLOR_BGR2GRAY)
		flow = cv2.calcOpticalFlowFarneback(prev_gray,next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0) #calc op
		
		next_frame = next_frame.astype('float32')/255.0
		model_input = np.concatenate((next_frame,flow),axis=2)
		model_input = model_input.transpose(2, 0, 1)
		model_input = model_input[None, ...]  
		model_input = np.ascontiguousarray(model_input)
		model_input = torch.from_numpy(model_input)
		test_loader = DataLoader(
					model_input,
					batch_size=1, shuffle=False,
					num_workers=0,
					pin_memory=True
					)
		attention_start = time.perf_counter()
		outputs = predict_saliency(test_loader, model)
		attention_end = time.perf_counter()
		attention_time = attention_end-start
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
		
		return mask_frame,flow_color,flow_time,attention_time


def do_opticalflow(prev_frame,next_frame):

	#calucu optical flow
	ori_frame = next_frame
	prev_frame =  cv2.resize(prev_frame, (320, 192), interpolation=cv2.INTER_CUBIC)
	prev_gray = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)

	next_frame =  cv2.resize(next_frame, (320, 192), interpolation=cv2.INTER_CUBIC)
	next_gray = cv2.cvtColor(next_frame,cv2.COLOR_BGR2GRAY)
	hsv = np.zeros_like(prev_frame)
	hsv[...,1] = 255
	flow = cv2.optflow.calcOpticalFlowSparseToDense(prev_gray,next_gray,sigma = 0.07) 
	#flow = cv2.calcOpticalFlowFarneback(prev_gray,next_gray, None, 0.5, 3, 25, 3, 5, 1.2, 0) #calc op with frane algorithm
	mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
	hsv[...,0] = ang*180/np.pi/2
	hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
	bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
	

	return bgr


def predict(model, test_data):#,test_data2):

	#predoct traffic risk assessment
	inputs = Variable(test_data).cuda()
	outputs = model(inputs) 
	outputs = F.softmax(outputs)
	#return classes[outputs.argmax()]
	return outputs

def predict_saliency(valid_loader, model):

	#predict driver attention

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



def object_detect(frame,model):

	#run object detection
	frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC) 
	yolo_img = frame
	yolo_img =  yolo_img[..., ::-1]
	results = model(yolo_img, size=224)
	results = results.pandas().xyxy[0]

	return results

def object_weight(blur,frame,results):

	# get object weight image
	frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)
	for index,row in results.iterrows():
		y =  int(row["ymin"])
		h =  int(row["ymax"])-int(row["ymin"])
		x = int(row["xmin"])
		w = int(row["xmax"])-int(row["xmin"])
		roi = frame[y:y+h,x:x+w]
		blur[y:y+h,x:x+w]=roi*0.7

	return blur


opt = parse_opts()
# ----------------QT set----------------
print(classes)
qt_env = QApplication(sys.argv)
process = Qt()
fileUrl = process.mv_Chooser()
print(fileUrl)
if(fileUrl == ""):
	print("Without input file!!")
	sys.exit(0)


#----------------model oupuy info set----------------
full_rgb_clip = []
full_op_clip = []
full_attention_clip=[]

distract_output = ""
font = cv2.FONT_HERSHEY_SIMPLEX

# ----------------get saliency model----------------
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.manual_seed(2017)
torch.cuda.manual_seed(2017)
random.seed(2017)
np.random.seed(2017)

attention_model = Model()
attention_model = attention_model.cuda()
params = attention_model.parameters()
cudnn.benchmark = True

#best = torch.load('./saliency/op.tar')  #my model
best = torch.load(opt.attention_path)
attention_model.load_state_dict(best['state_dict'])


#----------------get yolo model----------------
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
yolo_model.classes=[0,1,2,3,4,5,6,7]


# ----------------get assessment  model----------------
opt.mean = get_mean(1, dataset='kinetics')
opt.std = get_std(1)
#print('mean:', opt.mean)
norm_method = Normalize(opt.mean, [1, 1, 1])

if not opt.use_cbam:
	model_rgb, parameters = generate_model(opt,'ori')
	model_flow,parameters = generate_model(opt,'ori')

else:
	model_rgb, parameters = generate_model(opt,'cbam')
	model_flow,parameters = generate_model(opt,'cbam')



checkpoint_rgb = torch.load(opt.assess_rgb_path)
checkpoint_flow = torch.load(opt.assess_flow_path)
model_rgb.load_state_dict(checkpoint_rgb['state_dict'])
model_flow.load_state_dict(checkpoint_flow['state_dict'],strict=False)
model_flow.eval()
model_rgb.eval()

spatial_transform = Compose([
	letter_img(opt.sample_size),
	ToTensor(opt.norm_value), 
	norm_method
])
temporal_transform = TemporalCenterCrop(opt.sample_duration)



# ----------------get video----------------
flag=0
cap = cv2.VideoCapture(fileUrl)
print(fileUrl)
video_name = fileUrl.split('/') 
ret, frame = cap.read()
height, width = frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
critical_out = cv2.VideoWriter('./critical_clip/video_'+video_name[len(video_name)-1], fourcc, 30, (224,224))
video_out = cv2.VideoWriter('./test_video.mp4', fourcc, 30, (width,int(2*height)))


# ----------------output information----------------
critical_score=[]
high_score=[]
medium_score=[]
low_score=[]
time_frame=[]


flow_time = [] #flow calcu time
sa_time = [] #attention calcu time

cs,hs,ls,ms=0,0,0,0


obj_time=[]
attention_time=[]
assessment_time=[]
system_time=[]


num=0
risk =''
frame_num=1
# ----------------video image proscess----------------
while(ret):

	ret, frame = cap.read()
	if(not ret):
		break

	#first frame read
	if flag==0:
		flag=flag+1
		prev_frame = frame.copy()
		prev_obj_results = object_detect(frame,yolo_model)


	elif flag>0:

		resize_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC) 
		blurred = cv2.multiply(resize_frame,(0.4,0.4,0.4,0.4))

		#----------------object detection----------------
		start = time.perf_counter()
		obj_results = object_detect(frame,yolo_model)
		rgb_object_image = object_weight(blurred,frame,obj_results)
		obj_end = time.perf_counter()
		obj_end_time = obj_end-start
		obj_time.append(obj_end_time)

		#----------------driver attrntion----------------
		weight_frame,optical_frame,flowt,sat = do_saliency(prev_frame,frame,attention_model)
		flow_time.append(flowt)
		sa_time.append(sat)
		attention_end  = time.perf_counter()
		attention_end_time = attention_end-start
		attention_time.append(attention_end_time)


		optical_blur = optical_frame.copy()
		optical_blur = cv2.multiply(optical_blur,(0.4,0.4,0.4,0.4))
		op_object_image = object_weight(optical_blur,optical_frame,obj_results)
		op_object_image = object_weight(op_object_image,optical_frame,prev_obj_results)

		#----------------add object and attention weight to frame----------------
		rgb_object_image[weight_frame>50]= resize_frame[weight_frame>50] #rgb weighted image
		op_object_image[weight_frame>50] = optical_frame[weight_frame>50] #flow weighted image

		prev_frame = frame.copy()
		prev_obj_results = object_detect(frame,yolo_model)
		
		#----------------store serial weighted frame------------------------
		full_rgb_clip.append(rgb_object_image)
		full_op_clip.append(op_object_image)

		time_frame.append(frame_num)
		frame_num=frame_num+1

		#----------------traffic risk assessment----------------------
		if len(full_rgb_clip) > opt.assess_length:
			assess_start = time.perf_counter()
			test_data_rgb = get_test_data(full_rgb_clip, spatial_transform)
			test_data_op = get_test_data(full_op_clip, spatial_transform)

			rgb_distract_output = predict(model_rgb, test_data_rgb) #rgb image predict
			#print('rrr',rgb_distract_output)
			op_distract_output = predict(model_flow, test_data_op) #flow image predict
			#print('ooo',op_distract_output)
			assess_end = time.perf_counter()
			assess_end_time = assess_end-assess_start
			assessment_time.append(assess_end_time)

			risk_score = (rgb_distract_output+op_distract_output)/2 #fusion result
			risk_score = risk_score.cpu().detach().numpy()
			#print(risk_score)
			risk = classes[risk_score.argmax()] #the highest score class


			#store ouput score
			cs = risk_score[0][0]
			hs = risk_score[0][1]
			ls = risk_score[0][2]
			ms = risk_score[0][3]

			#sliding window or sequence			
			if opt.assess_input == 'sliding_window':
				full_rgb_clip.pop(0)
				full_op_clip.pop(0)

			elif opt.assess_input == 'sequence':
				full_rgb_clip=[]
				full_op_clip=[]
					

		end = time.perf_counter()
		end_time = end-start
		system_time.append(end_time)

		critical_score.append(cs)
		high_score.append(hs)
		low_score.append(ls)
		medium_score.append(ms)
		

		#draw score plot
		plt.plot(time_frame,critical_score,color=(1,0,0),label="critical")
		plt.plot(time_frame,high_score,color=(0,1,0),label="high")
		plt.plot(time_frame,medium_score,color=(1,1,0),label="medium")
		plt.plot(time_frame,low_score,color=(0,0,1),label="low")
		plt.xticks(fontsize=10)
		plt.yticks(fontsize=10)
		plt.xlabel("Frame",fontsize=15)
		plt.title("Traffic Assessment",fontsize=15)
		plt.legend(loc='upper right')
		plt.savefig("test1.png")
		plt.close()
		plot_img=cv2.imread('./test1.png')
		plot_img = cv2.resize(plot_img, (width,height), interpolation=cv2.INTER_CUBIC) 

		n = frame_num-1
		

		cv2.putText(frame,risk,(50,100), font, 1.4,(0,0,255),3,cv2.LINE_AA)
		output = cv2.vconcat([frame,plot_img])
		show_output = cv2.resize(output, (1080,720), interpolation=cv2.INTER_CUBIC)
		cv2.imshow("Traffic Assessment",show_output)
		cv2.imshow("Driver Attention",rgb_object_image)
		video_out.write(output)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			cap.release()
			cv2.destroyAllWindows()
			break
		



plt.plot(time_frame,critical_score,color=(1,0,0),label="critical")
plt.plot(time_frame,high_score,color=(0,1,0),label="high")
plt.plot(time_frame,medium_score,color=(1,1,0),label="medium")
plt.plot(time_frame,low_score,color=(0,0,1),label="low")
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel("Frame",fontsize=15)
plt.title("Traffic Assessment",fontsize=15)
plt.legend(loc='upper right')
plt.savefig("test1.png")
plot_img=cv2.imread('./test1.png')


# show calcu time overall
print('avg objtime:',sum(obj_time)/len(obj_time))
print('avg obj  fps:',1/(sum(obj_time)/len(obj_time)))
print('avg attention time:',sum(attention_time)/len(attention_time))
print('avg attention  fps:',1/(sum(attention_time)/len(attention_time)))
print('avg flow time:',sum(flow_time)/len(flow_time))
print('avg flow  fps:',1/(sum(flow_time)/len(flow_time)))
print('avg sa time:',sum(sa_time)/len(sa_time))
print('avg sa  fps:',1/(sum(sa_time)/len(sa_time)))
print('avg assess time:',sum(assessment_time)/len(assessment_time))
print('avg assess fps:',1/(sum(assessment_time)/len(assessment_time)))
print('avg system time:',sum(system_time)/len(system_time))
print('avg system fps:',1/(sum(system_time)/len(system_time)))



#clean video store
cap.release()
critical_out.release()
video_out.release()
