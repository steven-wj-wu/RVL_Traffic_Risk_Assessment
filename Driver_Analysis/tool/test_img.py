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
#from model_op_attention import Model
from data_load import ImageList
import random
import warnings
import logging
import numpy as np
import json
import cv2
from os import listdir


warnings.simplefilter("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('-g', '--gpu', default='0', type=str,
                    metavar='N', help='GPU NO. (default: 0)')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--split', default=0, type=int)
args = parser.parse_args()

name = 'traffic_net'
ckpts = 'ckpts/cdnn/'  #save model
if not os.path.exists(ckpts): os.makedirs(ckpts)

log_file = os.path.join(ckpts + "/train_log_%s.txt" % (name, ))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', filename=log_file)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logging.getLogger('').addHandler(console)


def get_heatmap(ori_img,mask_img):
    #heatmap=cv2.imread('dog_mask.jpg')
        #heatmap= cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
    overlay=ori_img.copy()
    alpha=0.2
    cv2.rectangle(overlay, (0, 0), (ori_img.shape[1], ori_img.shape[0]), (255, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, ori_img, 1-alpha, 0, ori_img)
    cv2.addWeighted(mask_img, alpha, ori_img, 1-alpha, 0, ori_img) # 将热度图覆盖到原图
        #cv2.imshow('test',img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    return ori_img




def main():
    #global args, best_score
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.manual_seed(2017)
    torch.cuda.manual_seed(2017)
    random.seed(2017)
    np.random.seed(2017)

    model = Model()
    model = model.cuda()
    params = model.parameters()

    cudnn.benchmark = True
    optimizer = torch.optim.Adam(params, args.lr,
                                weight_decay=args.weight_decay)

    
    #test_root =  './media/test/' 
    #test_imgs = listdir('./media/test/mask')   #file_name list

    video_list = listdir('./media/test/video')
    video_name = './media/test/video/'+video_list[0]
    vc = cv2.VideoCapture(video_name)

    #best = torch.load('./ckpts/attention_op.tar')  #my model
    best = torch.load('./op.tar')  #my model
    #best = torch.load('./ckpts/cdnn.tar')  #cdnn model

    model.load_state_dict(best['state_dict'])

    img1_path = './v_critical_g0010/frame000075.jpg'
    img2_path = './v_critical_g0010/frame000080.jpg'


    prev_frame =  cv2.imread(img1_path)
    prev_frame = cv2.resize(prev_frame, (320, 192), interpolation=cv2.INTER_CUBIC)
    prev_gray=cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)
    current_frame = cv2.imread(img2_path)
    current_frame = cv2.resize(current_frame, (320, 192), interpolation=cv2.INTER_CUBIC)
   
    next_gray=cv2.cvtColor(current_frame,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray,next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    ori_frame = current_frame.copy()
    current_frame = current_frame.astype('float32')/255.0
    model_input = np.concatenate((current_frame,flow),axis=2)
    model_input = model_input.transpose(2, 0, 1)
    model_input = model_input[None, ...]  
    model_input = np.ascontiguousarray(model_input)
    model_input = torch.from_numpy(model_input)

    test_loader = DataLoader(
            model_input,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers,
            pin_memory=True)
    outputs = predict(test_loader, model)
    mask = outputs[0]

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

    mask_img= cv2.applyColorMap(mask_img,cv2.COLORMAP_JET)
    heat_img_mask = get_heatmap(ori_frame,mask_img)
    cv2.imwrite('show3.jpg',heat_img_mask)



    cv2.waitKey(0)
    cv2.destroyAllWindows()




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






class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // (args.epochs//3)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
