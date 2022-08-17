import torch
from torch.autograd import Variable
import time
import sys
from apex import amp
#from utils import AverageMeter, calculate_accuracy
from utils import AverageMeter, calculate_accuracy_for_test
import time
import os
import numpy as np
from torchvision.utils import save_image
import torch.nn.functional as F
import numpy

def val_epoch(epoch, data_loader, model, criterion, opt, logger, best_acc, optimizer=None):
    print('validation at epoch {}'.format(epoch))


    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    n_correct_critical, n_correct_high, n_correct_low, n_correct_medium = 0., 0., 0., 0.
    num_critical, num_high, num_low, num_medium = 0., 0., 0., 0.
    ch,cl,cm=0.,0.,0.
    hc,hl,hm=0.,0.,0.
    lc,lh,lm=0.,0.,0.
    mc,mh,ml=0.,0.,0.
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        #print(type(inputs))
        #data_time = 0
        if not opt.no_cuda:
            #targets = targets.cuda(async=True)
            targets = targets.cuda()
        st = time.time()
        inputs = Variable(inputs).cuda()
        #inputs2 = Variable(inputs2, volatile=True).cuda()
        targets = Variable(targets).cuda()
        print(targets)
        outputs = model(inputs)
        #outputs = F.softmax(outputs)
        print(outputs)
        print('time: %.5f'%(time.time() - st))
        loss = criterion(outputs, targets)
        #acc = calculate_accuracy(outputs1, targets)
        print('-------------',targets)
        # n_<action>: number of the action in one batch
        # c_<action>: number of correctly classified action in one batch
        
        acc, n_critical, n_high, n_low, n_medium, correct_critical, correct_high, correct_low,correct_medium,critical_high,critical_low,critical_medium,high_critical,high_low,high_medium,low_crtical,low_high,low_medium,medium_critical,medium_high,medium_low = calculate_accuracy_for_test(outputs, targets)
        """
        num_sitting = num_sitting + n_sitting
        num_standing = num_standing + n_standing
        correct_sitting = correct_sitting + c_sitting
        correct_standing = correct_standing + c_standing
        """
        num_critical = num_critical+n_critical
        num_high = num_high+n_high
        num_low = num_low+n_low
        num_medium = num_medium+n_medium

        n_correct_critical = n_correct_critical+correct_critical
        n_correct_high = n_correct_high+correct_high
        n_correct_low = n_correct_low+correct_low
        n_correct_medium = n_correct_medium+correct_medium

        ch=ch+critical_high
        cl=cl+critical_low
        cm=cm+critical_medium
        hc=hc+high_critical
        hl=hl+high_low
        hm=hm+high_medium
        lc=lc+low_crtical
        lh=lh+low_high
        lm=lm+low_medium
        mc=mc+medium_critical
        mh=mh+medium_high
        ml=ml+medium_low

        
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  acc=accuracies))
        
    #input()
    
    logger.log({'epoch': epoch, 'loss': losses.avg,
    'acc': accuracies.avg,\
    'acc_critical': float(n_correct_critical/num_critical),\
    'acc_high': float(n_correct_high/num_high),\
    'acc_low': float(n_correct_low/num_low),\
    'acc_medium': float(n_correct_medium/num_medium),\
    'time': time.time()})
    
    #print('acc_sit:',correct_sit/num_sit, ' acc_stand:',correct_stand/num_stand)
    print('num c h l m', num_critical,'|',num_high,'|',num_low,'|' ,num_medium )
    print('acc_critical:',float(n_correct_critical/num_critical))
    print('acc_high:',float(n_correct_high/num_high))
    print('acc_low:',float(n_correct_low/num_low))
    print('acc_medium:',float(n_correct_medium/num_medium))
    print('critical_high:',float(ch/num_critical))
    print('critical_low:',float(cl/num_critical))
    print('critical_medium:',float(cm/num_critical))
    print('high_critical',float(hc/num_high))
    print('high_low:',float(hl/num_high))
    print('high_medium:',float(hm/num_high))
    print('low_crtical:',float(lc/num_low))
    print('low_high:',float(lh/num_low))
    print('low_medium:',float(lm/num_low))
    print('medium_critical:',float(mc/num_medium))
    print('medium_high:',float(mh/num_medium))
    print('medium_low:',float(ml/num_medium))
    print('acc:',accuracies.avg)

    

    acc = round(accuracies.avg,2)
   
    return losses.avg, acc


def val_epoch_two_stream(epoch, data_loader_rgb,data_loader_flow, model_rgb, model_flow, criterion, opt, logger, best_acc, optimizer=None):
    print('validation at epoch {}'.format(epoch))

    f = open('./output.txt', 'w')
    model_rgb.eval()
    model_flow.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    #num_sitting, num_standing, correct_sitting, correct_standing = 0., 0., 0., 0.,
    #num_sit, num_stand, correct_sit, correct_stand = 0., 0., 0., 0.,

    n_correct_critical, n_correct_high, n_correct_low, n_correct_medium = 0., 0., 0., 0.
    num_critical, num_high, num_low, num_medium = 0., 0., 0., 0.
    ch,cl,cm=0.,0.,0.
    hc,hl,hm=0.,0.,0.
    lc,lh,lm=0.,0.,0.
    mc,mh,ml=0.,0.,0.

    dataloader_iterator = iter(data_loader_flow)

    for i, (rgb_inputs, rgb_targets) in enumerate(data_loader_rgb):


        data_time.update(time.time() - end_time)
        t= rgb_targets.numpy()[0]
        #print(type(inputs))
        #data_time = 0
        if not opt.no_cuda:
            #targets = targets.cuda(async=True)
            rgb_targets = rgb_targets.cuda()

        st = time.time()

        try:
            flow_inputs,flow_targets = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(data_loader_flow)
            flow_inputs,flow_targets = next(dataloader_iterator)


        rgb_inputs = Variable(rgb_inputs).cuda()
        rgb_outputs = model_rgb(rgb_inputs)

        flow_inputs = Variable(flow_inputs).cuda()
        flow_outputs = model_flow(flow_inputs)

        targets = Variable(rgb_targets).cuda()
        print(targets)


        print('time: %.5f'%(time.time() - st))

        outputs = rgb_outputs

        #outputs = (rgb_outputs+flow_outputs)/2
        #soft_outputs = (F.softmax(rgb_outputs)+F.softmax(flow_outputs))/2
        soft_outputs = (rgb_outputs+flow_outputs)/2
        soft_outputs=F.softmax(soft_outputs)
        #print(F.softmax(rgb_outputs))
        #print(F.softmax(flow_outputs))
        ans = soft_outputs.argmax()
        print('num:'+str(i)+'target:'+str(t))
        f.write('num:'+str(i)+'target:'+str(t)+'predict:'+str(ans)+'\n')
        print(soft_outputs)


        loss = criterion(soft_outputs, targets)
        #acc = calculate_accuracy(outputs1, targets)
        print('-------------',targets)
        # n_<action>: number of the action in one batch
        # c_<action>: number of correctly classified action in one batch
        
        acc, n_critical, n_high, n_low, n_medium, correct_critical, correct_high, correct_low,correct_medium,critical_high,critical_low,critical_medium,high_critical,high_low,high_medium,low_crtical,low_high,low_medium,medium_critical,medium_high,medium_low = calculate_accuracy_for_test(soft_outputs, targets)
        """
        num_sitting = num_sitting + n_sitting
        num_standing = num_standing + n_standing
        correct_sitting = correct_sitting + c_sitting
        correct_standing = correct_standing + c_standing
        """
        num_critical = num_critical+n_critical
        num_high = num_high+n_high
        num_low = num_low+n_low
        num_medium = num_medium+n_medium

        n_correct_critical = n_correct_critical+correct_critical
        n_correct_high = n_correct_high+correct_high
        n_correct_low = n_correct_low+correct_low
        n_correct_medium = n_correct_medium+correct_medium

        ch=ch+critical_high
        cl=cl+critical_low
        cm=cm+critical_medium
        hc=hc+high_critical
        hl=hl+high_low
        hm=hm+high_medium
        lc=lc+low_crtical
        lh=lh+low_high
        lm=lm+low_medium
        mc=mc+medium_critical
        mh=mh+medium_high
        ml=ml+medium_low

        
        
        losses.update(loss.item(), rgb_inputs.size(0))
        accuracies.update(acc, rgb_inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader_rgb),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  acc=accuracies))
        
    #input()
    
    logger.log({'epoch': epoch, 'loss': losses.avg,
    'acc': accuracies.avg,\
    'acc_critical': float(n_correct_critical/num_critical),\
    'acc_high': float(n_correct_high/num_high),\
    'acc_low': float(n_correct_low/num_low),\
    'acc_medium': float(n_correct_medium/num_medium),\
    'time': time.time()})
    
    #print('acc_sit:',correct_sit/num_sit, ' acc_stand:',correct_stand/num_stand)
    print('num c h l m', num_critical,'|',num_high,'|',num_low,'|' ,num_medium )
    print('acc_critical:',float(n_correct_critical/num_critical))
    print('acc_high:',float(n_correct_high/num_high))
    print('acc_low:',float(n_correct_low/num_low))
    print('acc_medium:',float(n_correct_medium/num_medium))
    print('critical_high:',float(ch/num_critical))
    print('critical_low:',float(cl/num_critical))
    print('critical_medium:',float(cm/num_critical))
    print('high_critical',float(hc/num_high))
    print('high_low:',float(hl/num_high))
    print('high_medium:',float(hm/num_high))
    print('low_crtical:',float(lc/num_low))
    print('low_high:',float(lh/num_low))
    print('low_medium:',float(lm/num_low))
    print('medium_critical:',float(mc/num_medium))
    print('medium_high:',float(mh/num_medium))
    print('medium_low:',float(ml/num_medium))
    print('acc:',accuracies.avg)
    f.close()

    

    acc = round(accuracies.avg,2)
   
    return losses.avg, acc