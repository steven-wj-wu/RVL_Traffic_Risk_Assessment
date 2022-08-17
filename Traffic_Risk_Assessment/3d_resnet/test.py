import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
import sys
import json
import numpy as np
from utils import AverageMeter


def calculate_video_results(output_buffer, video_id, test_results, class_names,output_list):
    video_outputs = torch.stack(output_buffer)
    average_scores = torch.mean(video_outputs, dim=0)
    # For Winbus training (2 class)
    sorted_scores, locs = torch.topk(average_scores, k=1)
    print(np.shape(sorted_scores))
    print(average_scores)
    #print(sorted_scores, locs)
    #print(video_outputs)
    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({
            'label': class_names[torch.Tensor.item(locs[i])],
            'score': float(sorted_scores[i])
        })
    output_list += [str(video_id)+' '+str(class_names[torch.Tensor.item(locs[0])])]
    test_results['results'][video_id] = video_results


def test(data_loader, model, opt, class_names):
    print('test')
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end_time = time.time()
    output_buffer = []
    output_list = []
    previous_video_id = ''
    test_results = {'results': {}}
    for i, (inputs1,inputs2, targets) in enumerate(data_loader):
        #print(type(inputs))
        data_time.update(time.time() - end_time)
        st = time.time()
        # print('--------------------------------',inputs.shape)
        inputs1 = Variable(inputs1, volatile=True).cuda()
        inputs2 = Variable(inputs1, volatile=True).cuda()


        outputs1,outputs2 = model(inputs1,inputs2)
        outputs = (outputs1+outputs2)/2
        #print("--------------------??????", )
        if not opt.no_softmax_in_test:
            outputs = F.softmax(outputs)
        print('----------')
        #print(targets)
        #print(outputs)
        for j in range(outputs.size(0)):
            print('i=',i,' j=',j, ' targets=', targets[j], ' previous_video_id=', previous_video_id)
            if not (i == 0 and j == 0) and targets[j] != previous_video_id:
                calculate_video_results(output_buffer, previous_video_id,
                                        test_results, class_names,output_list)
                output_buffer = []
            output_buffer.append(outputs[j].data.cpu())
            previous_video_id = targets[j]
            print('------------',opt.test_subset)
        if (i % 100) == 0:
            with open(
                    os.path.join(opt.result_path, '{}.json'.format(
                        opt.test_subset)), 'w') as f:
                json.dump(test_results, f)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        #print('time:%.4f'%(time.time()-st))

        print('[{}/{}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time))
    out_file = open('output_data.txt','w')
    for out in output_list:
        out_file.write(out+'\n')
    #print(opt.result_path)
    #print('{}.json'.format(opt.test_subset))
    print('result = ',test_results)
    with open(
            os.path.join(opt.result_path, '{}.json'.format(opt.test_subset)),
            'w') as f:
        json.dump(test_results, f)
