import os
import sys
import cv2 
import numpy as np
from tqdm import tqdm
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import PyQt5.QtGui as QtGui
from PyQt5.QtCore import *
from pathlib import Path  

def read_classes(file_path):

	fp = open(file_path, "r")
	classes = fp.readline()
	classes = classes. split(",")
	fp.close()
	last = classes[1].split("\n")
	classes[1]=last[0]
	print(classes)

	return classes

class Qt(QWidget):
	def mv_Chooser(self):    
		opt = QFileDialog.Options()
		opt |= QFileDialog.DontUseNativeDialog
		fileUrl = QFileDialog.getOpenFileName(self,"Input Video", "/home/leo/Desktop/3dresnet_video/series_image/rgb_videos/1","AVI (*.avi)", options=opt)
	
		return fileUrl[0]


classes = read_classes('classes.txt')
print(classes)
num_frame = 10
imgSize = (224,224)

if __name__ == '__main__':

	#讀取影片路徑
	
	qt_env = QApplication(sys.argv)
	process = Qt()
	fileUrl = process.mv_Chooser()
	
	#fileUrl = "./001.mp4"
	print(fileUrl)

	if(not os.path.exists('./dataset')):
		for i in range(len(classes)):
			os.makedirs('./dataset/'+str(classes[i]))
		

	#開啟影片
	cap = cv2.VideoCapture(fileUrl)
	movie_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	
	#圖片處理
	
	ret, frame = cap.read()
	i = 1
	series = []

	while(ret):

		#frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
		#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#draw = gray.copy()
		#series.append(gray)
		draw = frame.copy()

		#調整訓練圖片大小
		frame = cv2.resize(frame, imgSize)
		series.append(frame)
		

		for c in range(len(classes)):
			cv2.putText(draw,str(c)+":"+classes[c],(20,20+c*40),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.2,(255),1)
		cv2.putText(draw,str(len(classes))+":"+"X",(20,20+len(classes)*40),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.2,(255),1)
		draw = cv2.resize(draw, (int(2*draw.shape[1]/3), int(2*draw.shape[0]/3)))
		cv2.imshow("src", draw)

		if(i%num_frame == 0):
			
			print('123123123123')
			class_index = input()
			if(class_index == 'q'):
				exit(0)
			class_index = int(class_index)
			if(class_index<len(classes)):
				path = "./dataset/"+str(classes[class_index])
				file_num = len([lists for lists in os.listdir(path) if os.path.isdir(os.path.join(path, lists))])
				print("---",file_num+7)
				os.makedirs(path+"/"+str(classes[class_index])+"_"+str(file_num).zfill(6))
				for n in range(num_frame):
					#存圖片時檔名數字須從1開始(3Dresnet從編號1開始讀取)
					
					path = "./dataset/"+str(classes[class_index])+"/"+str(classes[class_index])+"_"+str(file_num).zfill(6)+"/"+"image_"+str(n+1).zfill(5)+".jpg"
					cv2.imwrite(path, series[n])

			series = []
		if cv2.waitKey(1) & 0xFF == ord('q'):
			cap.release()
			cv2.destroyAllWindows()
			break
		i = i + 1
		ret, frame = cap.read()

	cap.release()
	cv2.destroyAllWindows()

	
	
		
		
