import cv2
import torch
from PIL import Image

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x')

model.classes=[0,1,2,3,4,5,6,7]


img = cv2.imread('image_00009.jpg')
ori_img = img.copy()
background = img.copy()
background = cv2.multiply(background,(0.5,0.5,0.5,0.5))


im2 = img[..., ::-1]  # OpenCV image (BGR to RGB)
imgs = [im2]  # batch of images

# Inference
results = model(imgs, size=224)  # includes NMS

# Results
results.print()  
#results.show()  # or .show()

results.xyxy[0]  # im1 predictions (tensor)
#print(results.xyxy[0])

result = results.pandas().xyxy[0]  # im1 predictions (pandas)

for index,row in result.iterrows():

	y =  int(row["ymin"])
	h =  int(row["ymax"])-int(row["ymin"])
	x = int(row["xmin"])
	w = int(row["xmax"])-int(row["xmin"])
	print("x:",int(row["xmin"]))
	print("w",int(row["xmax"])-int(row["xmin"]))
	print("y:",int(row["ymin"]))
	print("h",int(row["ymax"])-int(row["ymin"]))
	roi = ori_img[y:y+h,x:x+w]
	background[y:y+h,x:x+w]=roi

cv2.imshow("123",background)
cv2.waitKey(0)
cv2.destroyAllWindows()

	
			
					




