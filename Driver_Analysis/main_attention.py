import argparse
import os
import time
import pickle as pickle
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from model_op_attention import Model
from data_load_op import ImageList
import random
import warnings
import logging
import numpy as np
import json
from os import listdir


warnings.simplefilter("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=250, type=int, metavar='N',
                    help='numbeepochr of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('-g', '--gpu', default='0', type=str,
                    metavar='N', help='GPU NO. (default: 0)')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
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

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'],strict=False)
            #optimizer.load_state_dict(checkpoint['optim_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

            for idx,p in enumerate(model.convd1):
                if not (idx == 2 or idx == 5):
                    p.weight.requires_grad = False
            for idx,p in enumerate(model.convd2):
                if not (idx == 2 or idx == 5):
                    p.weight.requires_grad = False
            for idx,p in enumerate(model.convd3):
                if not (idx == 2 or idx == 5):
                    p.weight.requires_grad = False
            for idx,p in enumerate(model.convd4):
                if not (idx == 2 or idx == 5):
                    p.weight.requires_grad = False
            for idx,p in enumerate(model.convu3):
                if not (idx == 2 or idx == 5):
                    p.weight.requires_grad = False
            for idx,p in enumerate(model.convu2):
                if not (idx == 2 or idx == 5):
                    p.weight.requires_grad = False
            for idx,p in enumerate(model.convu1):
                if not (idx == 2 or idx == 5):
                    p.weight.requires_grad = False

            #model.upsample.weight.requires_grad = False
            #model.maxpool.weight.requires_grad = False
          



        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    train_root = './media/train/train_video/'  ### traffic_frames root
    valid_root = './media/valid/valid_15'
    test_root =  './media/test/'

    #train_imgs = [json.loads(line) for line in open(root + 'train.json')]
    #valid_imgs = [json.loads(line) for line in open(root + 'valid.json')]
    #test_imgs = [json.loads(line) for line in open(root + 'test.json')]
    # print len(train_imgs),train_imgs[0]
    # # print train_imgs
    # exit(0)
    
    train_imgs = listdir('./media/train/mask')  
    valid_imgs = listdir('./media/valid/mask_15')
    test_imgs = listdir('./media/test/mask')    
    #train_imgs = ['6_1.jpg','6_6.jpg','6_8.jpg']
    #valid_imgs = ['6_2.jpg']
    #test_imgs = ['6_14.jpg']
    


    train_loader = DataLoader(
            ImageList(train_root, train_imgs, for_train=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers,
            pin_memory=True)

    valid_loader = DataLoader(
            ImageList(valid_root, valid_imgs,for_train=False),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers,
            pin_memory=True)

 
    criterion = nn.BCELoss().cuda()

    logging.info('-------------- New training session, LR = %f ----------------' % (args.lr, ))
    logging.info('-- length of training images = %d--length of valid images = %d--' % (len(train_imgs),len(valid_imgs)))
    logging.info('-- length of test images = %d--' % (len(test_imgs)))
    best_loss = float('inf')
    file_name = os.path.join(ckpts, 'model_op_best_%s.tar' % (name, ))


    for epoch in range(args.start_epoch, args.epochs):
        
        #adjust_learning_rate(optimizer, epoch)
       

        # train for one epoch
        train_loss = train(
                train_loader, model, criterion, optimizer, epoch)

     
        print (train_loss)
      
        # evaluate on validation set
        valid_loss = validate(
                valid_loader, model, criterion)

        # remember best lost and save checkpoint
        best_loss = min(valid_loss, best_loss)
        if  epoch%20 ==  0 :
            file_name_last = os.path.join(ckpts, 'model_op_epoch_%d.tar' % (epoch + 1, ))
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                'valid_loss': valid_loss,
            }, file_name_last)

            if valid_loss == best_loss:
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss,
                }, file_name)


            msg = 'Epoch: {:02d} Train loss {:.4f} | Valid loss {:.4f}'.format(
                    epoch+1, train_loss, valid_loss)
            print(msg)
      
       
            logging.info(msg)



def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
 

    # switch to train mode
    model.train()
 
    start = time.time()

    #print(enumerate(train_loader))
  
    for i, (input, target) in enumerate(train_loader):

        input = input.cuda()
        target = target.cuda()
      

        #input = input.cuda(async=True)
        #target = target.cuda(async=True)

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
      

        # compute output
        output = model(input_var)

        # m = nn.Sigmoid()
        # loss = criterion(m(output), m(target_var))

        loss = criterion(output, target_var)

        # measure accuracy and record loss
        losses.update(loss.item(), target.size(0))

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if (i+1) % 100 == 0:
            msg = 'Training Epoch {:03d}  Iter {:03d} Loss {:.6f} in {:.3f}s'.format(epoch+1, i+1, losses.avg, time.time() - start)
            start = time.time()
            logging.info(msg)
            print(msg)

    return losses.avg

def validate(valid_loader, model, criterion):
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    start = time.time()
    for i, (input, target) in enumerate(valid_loader):
        input = input.cuda()
        target = target.cuda()


        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)

        loss = criterion(output, target_var)
        # measure accuracy and record loss
        losses.update(loss.item(), target.size(0))

        msg = 'Validating Iter {:03d} Loss {:.6f} in {:.3f}s'.format(i+1, losses.avg, time.time() - start)
        start = time.time()
        #logging.info(msg)
        print(msg)

    return losses.avg




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
