import os
import time
import numpy as np
import json
import cv2
from os import listdir



def get_heatmap(ori_img,mask_img):
	overlay=ori_img.copy()
	alpha=0.2
	cv2.rectangle(overlay, (0, 0), (ori_img.shape[1], ori_img.shape[0]), (255, 0, 0), -1)
	cv2.addWeighted(overlay, alpha, ori_img, 1-alpha, 0, ori_img)
	cv2.addWeighted(mask_img, alpha, ori_img, 1-alpha, 0, ori_img) 
	return ori_img




def main():

	ori_video='./video/897.mp4'
	mask_video='./video/897_pure_hm.mp4'

	ori_vc = cv2.VideoCapture(ori_video)
	mask_vc = cv2.VideoCapture(mask_video)
	print('123')
	fourcc = cv2.VideoWriter_fourcc(*'MP4V')
	out = cv2.VideoWriter('897_gt.mp4',fourcc, 30, (320,192))
	print('321')
	while(ori_vc.isOpened() and mask_vc.isOpened()):
		print('123')
		ret1,video_frame1 = ori_vc.read()
		ret2,video_frame2 = mask_vc.read()
		if ret1 and ret2:
			video_frame1 = cv2.resize(video_frame1, (320, 192), interpolation=cv2.INTER_CUBIC)
			video_frame2 = cv2.resize(video_frame2, (320, 192), interpolation=cv2.INTER_CUBIC)

			video_frame2 = cv2.applyColorMap(video_frame2,cv2.COLORMAP_JET)
			heat_img = get_heatmap(video_frame1,video_frame2)
			out.write(heat_img)
		else:
			break
		
	ori_vc.release()
	mask_vc.release()
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

		
