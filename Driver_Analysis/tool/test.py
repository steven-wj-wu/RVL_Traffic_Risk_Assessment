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

    
    test_root =  './media/test_small/' 
    test_imgs = listdir('./media/test_small/mask')	#file_name list

   

    print(test_imgs)

    
    start = time.time()
    test_loader = DataLoader(
            ImageList(test_root, test_imgs),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers,
            pin_memory=True)
    print(time.time() - start)

    criterion = nn.BCELoss().cuda()

   
    #file_name = os.path.join(ckpts, 'model_best_%s.tar' % (name, ))
    #checkpoint = torch.load(file_name)
    #model.load_state_dict(checkpoint['state_dict'])

    best = torch.load('./ckpts/best.tar')
    model.load_state_dict(best['state_dict'])

    c=0

    outputs, targets = predict(test_loader, model)
    
    """
    # for group test
    for img in outputs:
        mask = img*255
        true = targets[0]*255
        mask_img = np.zeros((192, 320,1), np.uint8)
        true_img = np.zeros((192, 320,1), np.uint8)
        col=0
        row=0
        print(mask)
	
        for i in mask:
            for j in i:
                mask_img[col,row,0]=j
                row=row+1
            row=0	
            col=col+1

        col=0
        row=0
        for i in true:
            for j in i:
                true_img[col,row,0]=j
                row=row+1
            row=0	
            col=col+1
        
        cv2.imshow("true", true_img)
        cv2.imshow("predict", mask_img)
        cv2.waitKey(0)
	cv2.destroyAllWindows()

	
        mask_img = cv2.resize(mask_img, (224, 224), interpolation=cv2.INTER_AREA)
        heat_img_mask = get_heatmap(ori_img,mask_img)
        cv2.imwrite('./test_write/'+test_imgs[c],img)
      """
    mask = outputs[0]
    
    if mask.max() != 0:
       mask = mask/mask.max()  #normalize
    
    true = targets[0]*255
    mask = mask*255

      
    
    true_img = np.zeros((192, 320,1), np.uint8)
    mask_img = np.zeros((192, 320,1), np.uint8)
    
    col=0
    row=0
    

    for i in mask:
        for j in i:
            mask_img[col,row,0]=j
            row=row+1
        row=0	
        col=col+1

    col=0
    row=0
    
    mask_img = cv2.resize(mask_img, (224, 224), interpolation=cv2.INTER_AREA)
    mask_img= cv2.applyColorMap(mask_img,cv2.COLORMAP_JET)

    
    for i2 in true:
        for j2 in i2:
            true_img[col,row,0]=j2
            row=row+1
        row=0	
        col=col+1
    
    true_img = cv2.resize(true_img, (224, 224), interpolation=cv2.INTER_AREA)	
    true_img= cv2.applyColorMap(true_img,cv2.COLORMAP_JET)
    

    origin_img = cv2.imread(test_root+test_imgs[0])
    print(test_root+test_imgs[0]) 
 

    #true_img = get_heatmap(origin_img,true_img)
    #mask_img = get_heatmap(origin_img,mask_img)
   
    cv2.imshow("true", true_img)
    cv2.imshow("mask", mask_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #np.savez(ckpts + 'test.npz', p=outputs, t=targets)
    #with open(ckpts + 'test.pkl', 'wb') as f:
        #pickle.dump(test_imgs, f)

    



def predict(valid_loader, model):

    # switch to evaluate mode
    model.eval()

    targets = []
    outputs = []

    for i, (input, target) in enumerate(valid_loader):
       
        targets.append(target.numpy().squeeze(1))
        
        input = input.cuda()

        input_var = torch.autograd.Variable(input, volatile=True)
        # compute output
        output = model(input_var)
        outputs.append(output.data.cpu().numpy().squeeze(1))

    targets = np.concatenate(targets)
    outputs = np.concatenate(outputs)
    return outputs, targets




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
