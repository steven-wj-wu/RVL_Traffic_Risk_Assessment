import os
import sys
import json
import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
import tensorboardX

from opts import parse_opts
from model_d import generate_model
from mean import get_mean, get_std
from spatial_transforms_winbus import (
    Compose, Normalize, RandomHorizontalFlip, ToTensor, RandomVerticalFlip, 
    ColorAugment)
from temporal_transforms import LoopPadding, TemporalRandomCrop, TemporalCenterCrop
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set, get_test_set,get_validation_set_flow
from utils import Logger
from train import train_epoch
from validation4winbus import val_epoch_two_stream , val_epoch                                                                                                                                                                                                             
#from validation import val_epoch
from test import test
from utils import init_seeds
import time
from collections import OrderedDict

from torchvision import transforms
import PIL
from PIL import Image, ImageOps

# To fixed scale of person
def letterbox(img, resize_size, mode='square'):
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

#Override transform's resize function(using letterbox)
class letter_img(transforms.Resize):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """

        #letterbox(img,self.size).save('out.bmp')
        return letterbox(img, self.size)
    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

# Can print the model structure
def model_info(model, report='summary'):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if report == 'full':
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients' % (len(list(model.parameters())), n_p, n_g))

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn() https://arxiv.org/pdf/1708.02002.pdf
    # i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=2.5)
    def __init__(self, loss_fcn, gamma=0.5, alpha=0.5):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        # If no use of pytorch original crossentropy (use label smooth ce)
        #self.reduction = loss_fcn.reduction
        #self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, input, target):
        loss = self.loss_fcn(input, target)
        loss *= self.alpha * (1.000001 - torch.exp(-loss)) ** self.gamma  # non-zero power for gradient stability
        # If no use of pytorch original crossentropy (use label smooth ce)
        '''
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
        '''
        # If use label smooth ce
        return loss.mean()

if __name__ == '__main__':
    opt = parse_opts()
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    print(opt)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    # Ori code
    torch.manual_seed(opt.manual_seed)

    # Initialize
    #init_seeds()
    #torch.manual_seed(opt.manual_seed)
    #torch.cuda.manual_seed(opt.manual_seed)
    #torch.backends.cudnn.deterministic = True
    np.random.seed(opt.manual_seed)
    random.seed(opt.manual_seed)

    if not opt.use_mix:
        model_rgb, parameters = generate_model(opt,'flow')
        model_flow,parameters = generate_model(opt,'flow')
    else:
        model, parameters, optimizer, scheduler = generate_model(opt)

    #criterion = nn.CrossEntropyLoss(weight=weights)
    criterion = nn.CrossEntropyLoss()
    #criterion = LabelSmoothingCrossEntropy()
    g = 2
    criterion = FocalLoss(criterion, g)

    if not opt.no_cuda:
        criterion = criterion.cuda()

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        print('mean:', opt.mean)
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)


    if not opt.no_val:
        print('norm:', opt.norm_value)

        spatial_transform = Compose([
            #Scale(opt.sample_size),
            #CenterCrop(opt.sample_size),
            #註解測試 letter_img(opt.sample_size),
            ToTensor(opt.norm_value), 
            norm_method
        ])
        
        #temporal_transform = LoopPadding(opt.sample_duration)
        #temporal_transform = TemporalRandomCrop(opt.sample_duration)
        temporal_transform = TemporalCenterCrop(opt.sample_duration)
        target_transform = ClassLabel()

        validation_data_rgb = get_validation_set(opt, spatial_transform, temporal_transform, target_transform)
        validation_data_flow = get_validation_set_flow(opt, spatial_transform, temporal_transform, target_transform)

       
        
        val_loader_rgb = torch.utils.data.DataLoader(
            validation_data_rgb,
            #batch_size=opt.batch_size,
            batch_size=1,
            shuffle=False,
            #shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=False)
       
        val_loader_flow = torch.utils.data.DataLoader(
            validation_data_flow,
            #batch_size=opt.batch_size,
            batch_size=1,
            shuffle=False,
            #shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=False)
        
        


        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'acc', 'acc_critical', 'acc_high',\
            'acc_low', 'acc_medium', 'time'])

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))

        #checkpoint_rgb = torch.load('./best_rgb_10.pth')
        checkpoint_rgb = torch.load('./best_new_crop.pth')
        checkpoint_flow = torch.load('./best_flow.pth')

        #assert opt.arch == checkpoint_flow['arch']

        #opt.begin_epoch = checkpoint['epoch']
        #model.load_state_dict(checkpoint['state_dict'])
        #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        if not opt.test_weight:
            #changed
            print('--------------------------------------------------------------------------------')
            print('---------------------------------------using model--------------------------')
            # print('--------------------------------------------------------------------------------')
            model_dict = model.state_dict()
            pretrain_dict = {k:v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrain_dict)
            opt.begin_epoch = checkpoint['epoch']
            model.load_state_dict(model_dict)
        else:
            """
            new_state_dict = OrderedDict()
            for k, v in checkpoint_rgb.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            """
            print('222222222222222222222222222222222222222222222222222222222')
      
            model_rgb.load_state_dict(checkpoint_rgb['state_dict'])
            #model_rgb.load_state_dict(new_state_dict)
            model_flow.load_state_dict(checkpoint_flow['state_dict'])
       

    print('run')
    best_acc = 0.


    #validation_loss, acc = val_epoch_two_stream(0, val_loader_rgb,val_loader_flow, model_rgb,model_flow, criterion, opt,val_logger, best_acc) #two_stream

    validation_loss, acc = val_epoch(0, val_loader_rgb, model_rgb, criterion, opt,val_logger, best_acc)




   
